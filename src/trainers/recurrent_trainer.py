import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from collections import defaultdict

from models.recurrent import RecurrentModel, RecurrentMode
from trainers.base_trainer import BaseTrainer
from utils.logging_utils import master_print
from utils.sharding_utils import maybe_shard_with_gradients


class RecurrentTrainer(BaseTrainer):

    model: RecurrentModel


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
            self.device,
        )

        for module in self.model.fast_modules():
            module.fast_log_lr.no_muon = True
            module.fast_m.no_muon = True
            module.fast_p_r.weight.no_muon = True
            module.fast_p_l.weight.no_muon = True


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
                    "embed_tokens", "lm_head"
                )
            ):
                embeddings.append(parameter)

            elif any(
                key in name for key in (
                    "fast", "embedding_norm", "bidirectional_head",
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
        if (
            "embeddings" not in self.config.trainer.multiple_optimizers
        ):
            parameters.pop("embeddings")

        return parameters


    def loss_and_lm_grad(
        self,
        lm_states: torch.FloatTensor,
        input_ids: torch.LongTensor,
        assistant_mask: torch.BoolTensor,
    ):
        assert (lm_states.shape[0]*lm_states.shape[1]) % self.config.trainer.num_logit_iterations == 0

        batch_size = lm_states.shape[0]
        num_iter = self.config.trainer.num_logit_iterations

        labels = input_ids[:, 1:]
        output_mask = assistant_mask[:, 1:].float()
        token_weights = (
            output_mask
            / output_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
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
        token_weights = maybe_shard_with_gradients(
            token_weights.reshape(-1, num_iter)
        )

        losses = []
        for i in range(num_iter):

            logits = self.model.lm_head(
                lm_states_leaf[:, i]
            ).float()

            loss = F.cross_entropy(
                logits,
                labels[:, i].contiguous(),
                reduction="none",
            )
            loss = (loss * token_weights[:, i]).sum()

            losses.append(loss.detach())
            loss.backward()

            xm.optimization_barrier_([lm_states_leaf.grad])

        loss = torch.stack(losses).sum()

        lm_grad = lm_states_leaf.grad.reshape(
            lm_states.shape
        ).detach().to(lm_states.dtype)
        
        return loss, lm_grad
    

    @torch_xla.compile(full_graph=True)
    def first_pass(
        self,
        input_ids,
        assistant_mask,
        pad_mask,
        no_slow_grads=True,
        second_state_update=False,
    ):
        
        self.model.set_mode(RecurrentMode.TRAIN_FIRST)

        with self._autocast():

            hidden_states = self.model.forward_backbone(
                input_ids,
            )
            lm_states = self.model.model.norm(hidden_states[:, :-1])
            loss, lm_grad = self.loss_and_lm_grad(
                lm_states,
                input_ids,
                assistant_mask,
            )

            with torch.no_grad():
                embeddings = self.model.forward_embeddings(
                    hidden_states.detach(),
                    pad_mask,
                )

        if no_slow_grads and not self.model.disable_fast_weights:
            torch.autograd.backward(
                lm_states,
                lm_grad,
                inputs=(self.model.grad_buffers())
            )
        else:
            torch.autograd.backward(
                lm_states, lm_grad
            )

        if second_state_update:
            self.model.set_mode(RecurrentMode.TRAIN_SECOND)
        with self._autocast():
            self.model.update_state(
                embeddings,
                pad_mask,
            )
        
        return loss
    

    @torch_xla.compile(full_graph=True)
    def second_pass(
        self,
        input_ids,
        assistant_mask,
        pad_mask,
    ):

        with self._autocast():

            self.model.set_mode(RecurrentMode.INFERENCE)
            with torch.no_grad():
                hidden_states = self.model.forward_backbone(
                    input_ids,
                )
            embeddings = self.model.forward_embeddings(
                hidden_states.detach(),
                pad_mask,
            )

            self.model.set_mode(RecurrentMode.TRAIN_SECOND)
            hidden_states = self.model.forward_backbone(
                input_ids, embeddings, pad_mask
            )
            lm_states = self.model.model.norm(hidden_states[:, :-1])

            loss, lm_grad = self.loss_and_lm_grad(
                lm_states,
                input_ids,
                assistant_mask,
            )

        torch.autograd.backward(
            lm_states, lm_grad
        )

        with self._autocast():
            self.model.update_state(
                embeddings,
                pad_mask,
            )
        
        return loss


    @torch_xla.compile(full_graph=True)
    def post_forward(self):

        self.model.empty_state()

        grad_norm = self.clip_gradients()
        aux = self.optimization_step()

        self.model.zero_grad(set_to_none=False)

        return aux, grad_norm


    def train_step(self, batch):
        input_ids: torch.LongTensor = batch["input_ids"]
        assistant_mask: torch.BoolTensor = batch["assistant_mask"]
        pad_mask: torch.BoolTensor = batch["attention_mask"]

        episodes = tuple(zip(
            input_ids.unbind(dim=1),
            assistant_mask.unbind(dim=1),
            pad_mask.unbind(dim=1),
        ))
        terminal_index = len(episodes) - 1
        losses = []
        aux = {}

        # first loop
        for index, episode in enumerate(episodes):

            loss = self.first_pass(*episode)

            aux[f"lm_loss/episode_{index:02d}"] = loss
            losses.append(loss)

            master_print(
                f"First pass {index:02d} completed."
            )

        self.model.finalize_state()
        self.model.zero_grad(set_to_none=False)

        # second loop
        for index, episode in enumerate(episodes[:-1]):

            self.second_pass(*episode)
        
            master_print(
                f"Second pass {index:02d} completed."
            )

        # lm loss only for last slow gradds
        self.first_pass(
            *episodes[-1],
            no_slow_grads=False,
            second_state_update=True
        )
        aux["relative_grad_error"] = self.model.relative_grad_error()
        master_print(
            f"Second pass {terminal_index:02d} completed."
        )

        # optimizer step
        post_aux, grad_norm = self.post_forward()
        aux.update(post_aux)
        master_print("Optimization step completed.")

        # metricsc
        final_loss = torch.stack(losses).mean()
        aux["atom_count"] = pad_mask.long().sum()

        decades = defaultdict(list)
        for key, value in aux.items():

            if "episode_" not in key or key.endswith("00"):
                continue

            decade = int(key.split("_")[-1][0])
            decades[decade].append(value)

        for decade, values in decades.items():
            aux[
                f"grouped_lm_loss/decade_{decade:02d}"
            ] = torch.stack(values).mean()

        return final_loss, aux, grad_norm
