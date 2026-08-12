import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from collections import defaultdict
import numpy as np

from models.llama import LlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.logging_utils import master_print
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import scale_gradient


def split_given_size(n, size):
    arr = np.array_split(
        np.arange(n),
        np.arange(size,n,size)
    )
    return [a.tolist() for a in arr]


class LMHorizonTrainer(BaseTrainer):

    model: LlamaForCausalLM


    def post_init(self):

        if (
            self.config.trainer.use_autocast and
            "embeddings" not in self.config.trainer.multiple_optimizers
        ):
            # The frozen vocabulary projection is already cast to BF16 by
            # autocast for every use. Store it in that compute dtype to avoid
            # retaining both a replicated FP32 weight and its BF16 cast.
            self.model.lm_head.to(dtype=torch.bfloat16)


    def _autocast(self):
        return torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        )
    

    def get_trainable_parameters(self, model):

        slow = []
        embeddings = []
        for name, parameter in model.named_parameters():

            if any(
                key in name for key in (
                    "embed_tokens", "lm_head",
                )
            ):
                embeddings.append(parameter)

            else:
                slow.append(parameter)

        parameters = {
            "slow": slow,
            "embeddings": embeddings,
        }
        
        out = {}
        for k, v in parameters.items():
            if k in self.config.trainer.multiple_optimizers:
                out[k] = v

        return out


    def loss_and_lm_grad(
        self,
        lm_states: torch.FloatTensor,
        input_ids: torch.LongTensor,
        assistant_mask: torch.BoolTensor,
        valid_mask: torch.BoolTensor,
        gradient_scale: float = 1.0,
    ):
        batch_size, seq_len, _ = lm_states.shape
        num_iter = self.config.trainer.num_logit_iterations
        assert (batch_size * seq_len) % num_iter == 0

        labels = input_ids[:, 1:]
        assistant_mask = assistant_mask[:, 1:].float()
        valid_mask = valid_mask[:, 1:].float()
        aux_mask = valid_mask - assistant_mask

        assistant_weights = (
            assistant_mask
            / assistant_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            / batch_size
        )
        aux_weights = (
            aux_mask
            / aux_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            / batch_size
        )

        lm_states_leaf = maybe_shard_with_gradients(
            lm_states.detach().reshape(
                -1, num_iter, lm_states.shape[-1]
            )
        ).detach().requires_grad_(True)

        labels = maybe_shard_with_gradients(
            labels.reshape(-1, num_iter)
        )
        assistant_weights = maybe_shard_with_gradients(
            assistant_weights.reshape(-1, num_iter)
        )
        aux_weights = maybe_shard_with_gradients(
            aux_weights.reshape(-1, num_iter)
        )

        assistant_loss_parts = []
        aux_loss_parts = []
        for i in range(num_iter):

            logits = self.model.lm_head(
                lm_states_leaf[:, i]
            ).float()
            logits = scale_gradient(logits, gradient_scale)

            raw_loss = F.cross_entropy(
                logits,
                labels[:, i].contiguous(),
                reduction="none",
            )
            assistant_loss_part = raw_loss * assistant_weights[:, i]
            aux_loss_part = raw_loss * aux_weights[:, i]

            assistant_loss = assistant_loss_part.sum()
            aux_loss = aux_loss_part.sum()

            loss = assistant_loss + self.config.trainer.aux_loss_weight * aux_loss

            loss.backward()
            xm.optimization_barrier_([lm_states_leaf.grad])

            assistant_loss_parts.append(assistant_loss_part.detach())
            aux_loss_parts.append(aux_loss_part.detach())

        # Stacking the iteration columns reverses the reshape above and restores
        # the original token order. The weights include 1 / batch_size for the
        # backward reduction, so remove that factor to report one loss per input
        # sequence. Keeping this dimension is what lets callers recover the
        # individual episode losses after concatenating episodes along batch.
        assistant_loss = (
            torch.stack(assistant_loss_parts, dim=1)
            .reshape(batch_size, seq_len)
            .sum(dim=-1)
            * batch_size
        )
        aux_loss = (
            torch.stack(aux_loss_parts, dim=1)
            .reshape(batch_size, seq_len)
            .sum(dim=-1)
            * batch_size
        )

        lm_grad = lm_states_leaf.grad.reshape(
            lm_states.shape
        ).detach().to(lm_states.dtype)
        
        return assistant_loss, aux_loss, lm_grad


    def inner_step(
        self,
        input_ids,
        assistant_mask,
        valid_mask,
        n,
    ):

        with self._autocast():

            lm_states, _ = self.model.forward(
                input_ids=input_ids,
                shift_states=True,
                compute_logits=False
            )
            loss, aux_loss, lm_grad = self.loss_and_lm_grad(
                lm_states,
                input_ids,
                assistant_mask,
                valid_mask,
                gradient_scale=n,
            )

        torch.autograd.backward(
            lm_states, lm_grad
        )

        return loss, aux_loss


    @torch_xla.compile(full_graph=True)
    def multi_inner_step(
        self, *episodes
    ):
        input_ids, assistant_mask, valid_mask = zip(*episodes)
        n = len(input_ids)

        input_ids = maybe_shard_with_gradients(
            torch.cat(input_ids, dim=0)
        )
        assistant_mask = maybe_shard_with_gradients(
            torch.cat(assistant_mask, dim=0)
        )
        valid_mask = maybe_shard_with_gradients(
            torch.cat(valid_mask, dim=0)
        )

        losses, aux_losses = self.inner_step(
            input_ids,
            assistant_mask,
            valid_mask,
            n,
        )

        losses = list(losses.reshape(n, -1).mean(dim=1).unbind())
        aux_losses = list(aux_losses.reshape(n, -1).mean(dim=1).unbind())

        return losses, aux_losses


    @torch_xla.compile(full_graph=True)
    def post_forward(self):

        num_none_grad = len([p for p in self.model.parameters() if p.grad is None])

        grad_norm = self.clip_gradients()
        aux = self.optimization_step()

        self.model.zero_grad(set_to_none=False)

        aux["num_none_grad"] = num_none_grad

        return aux, grad_norm


    def train_step(self, batch):
        input_ids: torch.LongTensor = batch["input_ids"]
        assistant_mask: torch.BoolTensor = batch["assistant_mask"]
        valid_mask: torch.BoolTensor = batch["attention_mask"]

        episodes = tuple(zip(
            input_ids.unbind(dim=1),
            assistant_mask.unbind(dim=1),
            valid_mask.unbind(dim=1),
        ))
        assert len(episodes) > 1

        losses = []
        aux_losses = []
        aux = {}

        # first loop
        inds_list = split_given_size(
            len(episodes), self.config.trainer.num_episodes_per_call
        )
        for inds in inds_list:

            curr_losses, curr_aux_losses = self.multi_inner_step(
                *[episodes[i] for i in inds]
            )
            torch_xla.sync(wait=True)

            for i in range(len((curr_losses))):
                aux[f"lm_loss/episode_{inds[i]:02d}"] = curr_losses[i].detach()
                aux[f"aux_loss/episode_{inds[i]:02d}"] = curr_aux_losses[i].detach()

                losses.append(curr_losses[i].detach())
                aux_losses.append(curr_aux_losses[i].detach())

            master_print(
                f"First  pass {inds[-1]:02d} completed."
            )

        # optimizer step
        post_aux, grad_norm = self.post_forward()
        torch_xla.sync(wait=True)

        aux.update(post_aux)
        master_print("Optimization step completed.")

        # metrics
        final_loss = (
            torch.stack(losses).mean() +
            self.config.trainer.aux_loss_weight * torch.stack(aux_losses).mean()
        )
        aux["total_loss"] = torch.stack(losses).mean()
        aux["total_aux_loss"] = torch.stack(aux_losses).mean()
        aux["atom_count"] = valid_mask.long().sum()

        decades = defaultdict(list)
        aux_decades = defaultdict(list)
        for key, value in aux.items():

            if "episode_" not in key or key.endswith("00"):
                continue

            decade = int(key.split("_")[-1][0])

            if "aux" in key:
                aux_decades[decade].append(value)
            else:
                decades[decade].append(value)

        for decade, values in decades.items():
            aux[
                f"grouped_lm_loss/decade_{decade:02d}"
            ] = torch.stack(values).mean()
        for decade, values in aux_decades.items():
            aux[
                f"grouped_aux_loss/decade_{decade:02d}"
            ] = torch.stack(values).mean()

        return final_loss, aux, grad_norm
