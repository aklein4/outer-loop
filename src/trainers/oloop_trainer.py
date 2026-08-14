import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from collections import defaultdict

from trainers.base_trainer import BaseTrainer
from models.oloop import OLoopModel
from utils.logging_utils import master_print
from utils.sharding_utils import maybe_shard_with_gradients


class OLoopTrainer(BaseTrainer):
    
    model: OLoopModel


    def post_init(self):

        if (
            self.config.trainer.use_autocast and
            "embeddings" not in self.config.trainer.multiple_optimizers
        ):
            # The frozen vocabulary projection is already cast to BF16 by
            # autocast for every use. Store it in that compute dtype to avoid
            # retaining both a replicated FP32 weight and its BF16 cast.
            self.model.lm_head.to(dtype=torch.bfloat16)

        self.model.init_state(
            self.global_batch_size,
            self.device
        )


    def _autocast(self):
        return torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        )
    

    def get_trainable_parameters(self, model):

        slow = []
        fast = []
        embeddings = []
        for name, parameter in model.named_parameters():

            if any(
                key in name for key in (
                    "embed_tokens", "lm_head",
                )
            ):
                embeddings.append(parameter)

            elif any(
                key in name for key in (
                    "fast",
                )
            ):
                fast.append(parameter)

            else:
                slow.append(parameter)

        parameters = {
            "slow": slow,
            "fast": fast,
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

        if num_iter == 1:
            lm_states_leaf = lm_states.detach().requires_grad_(True)

            logits = self.model.lm_head(
                lm_states_leaf
            ).float()

            raw_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1).contiguous(),
                reduction="none",
            ).reshape(labels.shape)
            assistant_loss = (raw_loss * assistant_weights).sum()
            aux_loss = (raw_loss * aux_weights).sum()

            loss = assistant_loss + self.config.trainer.aux_loss_weight * aux_loss
            loss.backward()

            return assistant_loss.detach(), aux_loss.detach(), lm_states_leaf.grad.detach().to(lm_states.dtype)

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

        assistant_losses = []
        aux_losses = []
        for i in range(num_iter):

            logits = self.model.lm_head(
                lm_states_leaf[:, i]
            ).float()

            raw_loss = F.cross_entropy(
                logits,
                labels[:, i].contiguous(),
                reduction="none",
            )
            assistant_loss = (raw_loss * assistant_weights[:, i]).sum()
            aux_loss = (raw_loss * aux_weights[:, i]).sum()

            loss = assistant_loss + self.config.trainer.aux_loss_weight * aux_loss

            loss.backward()
            xm.optimization_barrier_([lm_states_leaf.grad])

            assistant_losses.append(assistant_loss.detach())
            aux_losses.append(aux_loss.detach())

        assistant_loss = torch.stack(assistant_losses).sum()
        aux_loss = torch.stack(aux_losses).sum()

        lm_grad = lm_states_leaf.grad.reshape(
            lm_states.shape
        ).detach().to(lm_states.dtype)
        
        return assistant_loss, aux_loss, lm_grad
    

    @torch_xla.compile(full_graph=True)
    def inner_step(
        self,
        input_ids,
        assistant_mask,
        valid_mask,
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
                valid_mask
            )

        torch.autograd.backward(
            lm_states, lm_grad
        )

        # self.model.update_state()

        return loss, aux_loss


    @torch_xla.compile(full_graph=True)
    def post_forward(self):

        num_none_grad = len([p for p in self.model.parameters() if p.grad is None])

        self.model.empty_state()

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
        for i, episode in enumerate(episodes):

            loss, aux_loss = self.inner_step(
                *episode
            )
            torch_xla.sync(wait=True)

            aux[f"lm_loss/episode_{i:02d}"] = loss.detach()
            aux[f"aux_loss/episode_{i:02d}"] = aux_loss.detach()

            losses.append(loss.detach())
            aux_losses.append(aux_loss.detach())

            master_print(
                f"Inner  pass {i:02d} completed."
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
