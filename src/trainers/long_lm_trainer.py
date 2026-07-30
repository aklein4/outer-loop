import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla
import torch_xla.core.xla_model as xm

from collections import defaultdict
import numpy as np

from models.llama import LlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.logging_utils import master_print
from utils.sharding_utils import maybe_shard_with_gradients


class LongLMTrainer(BaseTrainer):
    
    model: LlamaForCausalLM


    def post_init(self):

        self.model.model.embed_tokens.weight.no_muon = True
        try:
            self.model.lm_head.weight.no_muon = True
        except AttributeError:
            self.model.lm_head._orig_mod.weight.no_muon = True


    def _autocast(self):
        return torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        )


    def loss_and_lm_grad(
        self,
        lm_states: torch.FloatTensor,
        input_ids: torch.LongTensor,
    ):
        num_iter = self.config.trainer.num_logit_iterations
        assert (lm_states.shape[0]*lm_states.shape[1]) % self.config.trainer.num_logit_iterations == 0

        labels = input_ids[:, 1:]

        lm_states_leaf = maybe_shard_with_gradients(
            lm_states.detach().reshape(
                -1, num_iter, lm_states.shape[-1]
            )
        ).detach().requires_grad_(True)

        labels = maybe_shard_with_gradients(
            labels.reshape(-1, num_iter)
        )

        losses = []
        denom = labels.numel()
        for i in range(num_iter):

            with self._autocast():

                logits = self.model.apply_head(
                    lm_states_leaf[:, i, :]
                )

                loss = F.cross_entropy(
                    logits,
                    labels[:, i].contiguous(),
                    reduction="none",
                )

                losses.append(loss.detach())
                loss_for_backward = (
                    loss.sum() / denom
                ) / self.config.trainer.num_grad_accum_steps

                loss_for_backward.backward()
                xm.optimization_barrier_([lm_states_leaf.grad])

        loss = torch.stack(losses).mean()

        lm_grad = maybe_shard_with_gradients(
            lm_states_leaf.grad.reshape(*lm_states.shape)
        ).detach().to(lm_states.dtype)

        losses = torch.stack(losses, dim=1)
        losses = maybe_shard_with_gradients(
            losses.reshape(*lm_states.shape[:-1])
        )

        aux = {}
        chunks = torch.split(losses, self.config.trainer.chunk_size, dim=-1)
        for i, chunk in enumerate(chunks):
            aux[f"lm_loss/chunk_{i:02d}"] = chunk.mean()
        
        return loss, lm_grad, aux


    @torch_xla.compile(full_graph=True)
    def grad_accum(self, input_ids):

        with self._autocast():
            lm_states = self.model.forward(
                input_ids,
                shift_states=True,
                compute_logits=False
            )[0]

        loss, lm_grad, aux = self.loss_and_lm_grad(
            lm_states, input_ids
        )

        loss_for_backward = (lm_states * lm_grad.detach()).sum()
        loss_for_backward.backward()
        
        return loss, aux


    @torch_xla.compile(full_graph=True)
    def post_forward(self):

        num_none_grad = len([p for p in self.model.parameters() if p.grad is None])

        # regular optimization step
        grad_norm = self.clip_gradients()
        aux = self.optimization_step()
        self.model.zero_grad(set_to_none=False)

        aux["num_none_grad"] = num_none_grad

        return aux, grad_norm


    def train_step(self, batch):
        # TODO: we currently assume the batch has no padding

        input_ids: torch.LongTensor = batch["input_ids"]

        batches = torch.chunk(
            input_ids, self.config.trainer.num_grad_accum_steps,
            dim=0
        )
        batches = [
            maybe_shard_with_gradients(b) for b in batches
        ]

        losses = []
        auxes = []
        for i, b in enumerate(batches):

            loss, aux = self.grad_accum(b)
            losses.append(loss)
            auxes.append(aux)

            master_print(f"Minibatch {i:02d} completed.")

        loss = torch.stack(losses).mean()
        aux = {}
        for k in auxes[0].keys():
            if isinstance(auxes[0][k], torch.Tensor):
                aux[k] = torch.stack([a[k] for a in auxes]).mean()
            else:
                aux[k] = np.mean([a[k] for a in auxes])

        post_aux, grad_norm = self.post_forward()
        aux.update(post_aux)
        master_print("Optimization step completed.")

        # metrics
        aux["atom_count"] = input_ids.numel()

        decades = defaultdict(list)
        for key, value in aux.items():

            if "chunk_" not in key or key.endswith("00"):
                continue

            decade = int(key.split("_")[-1][0])
            decades[decade].append(value)

        for decade, values in decades.items():
            aux[
                f"grouped_lm_loss/decade_{decade:02d}"
            ] = torch.stack(values).mean()

        return loss, aux, grad_norm
