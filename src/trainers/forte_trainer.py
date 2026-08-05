import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from collections import defaultdict

from models.forte import ForteModel, ForteMode
from trainers.base_trainer import BaseTrainer
from utils.logging_utils import master_print
from utils.sharding_utils import (
    maybe_shard_no_gradients,
    maybe_shard_with_gradients,
)
from utils.torch_utils import ScannedTrainingLoop


class ForteTrainer(BaseTrainer):

    model: ForteModel


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
        self.first_pass_scan = ScannedTrainingLoop(
            None,
            self.scanned_first_pass,
        )
        self.second_pass_scan = ScannedTrainingLoop(
            self.model,
            self.scanned_second_pass,
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
                    "fast", "embedding_norm", "bidirectional_head", "embedding_state",
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
        if "embeddings" not in self.config.trainer.multiple_optimizers:
            parameters.pop("embeddings")
        if "slow" not in self.config.trainer.multiple_optimizers:
            parameters.pop("slow")

        return parameters


    def loss_and_lm_grad(
        self,
        lm_states: torch.FloatTensor,
        input_ids: torch.LongTensor,
        assistant_mask: torch.BoolTensor,
    ):
        batch_size, seq_len, _ = lm_states.shape
        num_iter = self.config.trainer.num_logit_iterations
        assert (batch_size * seq_len) % num_iter == 0

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


    def _scan_tensors(self, episodes):
        # Recurrence axes stay replicated; only the batch axis is partitioned.
        episode_spec = (None, ("data", "fsdp"), None)
        inputs = tuple(
            maybe_shard_no_gradients(torch.stack(values), spec=episode_spec)
            for values in zip(*episodes)
        )
        fast_modules = self.model.fast_modules()
        fast_spec = (None, ("data", "fsdp"), None, None)
        # Preserve these tensor identities inside the body: XLA uses them to
        # associate body outputs with While-loop carry slots.
        fast_states, fast_grad_buffers = (
            maybe_shard_no_gradients(
                torch.stack([getattr(mlp, name) for mlp in fast_modules]).detach(),
                spec=fast_spec,
            ).requires_grad_(True)
            for name in ("state", "grad_buffer")
        )
        return inputs, fast_modules, fast_states, fast_grad_buffers


    @staticmethod
    @torch.no_grad()
    def _store_fast_state(fast_modules, fast_states, fast_grad_buffers):
        for index, mlp in enumerate(fast_modules):
            mlp.state.copy_(fast_states[index])
            mlp.grad_buffer.copy_(fast_grad_buffers[index])
            mlp.state.grad.zero_()
            mlp.grad_buffer.grad.zero_()


    def scanned_first_pass(
        self,
        input_ids,
        assistant_mask,
        pad_mask,
        episode_slot,
        fast_states,
        fast_grad_buffers,
        episode_losses,
    ):
        """Pure first-pass body for an XLA While loop."""
        fast_states_leaf = fast_states
        fast_grad_buffers_leaf = fast_grad_buffers

        with self._autocast():
            with torch.no_grad():
                infer_hidden_states = self.model.forward_backbone(
                    input_ids,
                    mode=ForteMode.INFERENCE,
                    fast_states=fast_states_leaf,
                    fast_grad_buffers=fast_grad_buffers_leaf,
                )
                embeddings = self.model.forward_embeddings(
                    infer_hidden_states,
                    pad_mask,
                )

            hidden_states = self.model.forward_backbone(
                input_ids,
                mode=ForteMode.TRAIN_FIRST,
                embeddings=embeddings,
                embedding_mask=pad_mask,
                fast_states=fast_states_leaf,
                fast_grad_buffers=fast_grad_buffers_leaf,
            )
            lm_states = self.model.forward_lm_states(
                hidden_states,
                mode=ForteMode.TRAIN_FIRST,
                logits_to_keep=slice(0, -1),
                embeddings=embeddings,
                embedding_mask=pad_mask,
                fast_states=fast_states_leaf,
                fast_grad_buffers=fast_grad_buffers_leaf,
            )
            loss, lm_grad = self.loss_and_lm_grad(
                lm_states,
                input_ids,
                assistant_mask,
            )

        state_updates, raw_gradients = torch.autograd.grad(
            lm_states,
            (fast_states_leaf, fast_grad_buffers_leaf),
            lm_grad,
        )
        next_fast_states, next_fast_grad_buffers = (
            self.model.functional_update_state(
                fast_states_leaf,
                fast_grad_buffers_leaf,
                state_updates,
                raw_gradients,
                ForteMode.TRAIN_FIRST,
            )
        )
        next_episode_losses = episode_losses + episode_slot * loss.detach()

        # gradient_accumulation needs a differentiable scalar, but the first
        # sweep intentionally computes no slow-parameter gradients.
        return (
            loss,
            None,
            next_fast_states.detach(),
            next_fast_grad_buffers.detach(),
            next_episode_losses,
        )


    def scan_first_passes(self, episodes):
        (
            (input_ids, assistant_mask, pad_mask),
            fast_modules,
            fast_states,
            fast_grad_buffers,
        ) = self._scan_tensors(episodes)

        num_episodes = len(episodes)
        episode_slots = torch.eye(
            num_episodes,
            device=input_ids.device,
            dtype=torch.float32,
        )
        episode_slots = maybe_shard_no_gradients(
            episode_slots,
            spec=(None, None),
        )
        episode_losses = input_ids.new_zeros(
            num_episodes,
            dtype=torch.float32,
        )
        episode_losses = maybe_shard_no_gradients(
            episode_losses,
            spec=(None,),
        )

        _, fast_states, fast_grad_buffers, episode_losses = (
            self.first_pass_scan(
                (
                    input_ids,
                    assistant_mask,
                    pad_mask,
                    episode_slots,
                ),
                fast_states,
                fast_grad_buffers,
                episode_losses,
            )
        )

        self._store_fast_state(
            fast_modules, fast_states, fast_grad_buffers,
        )
        return episode_losses


    @torch_xla.compile(full_graph=True)
    def terminal_pass(
        self,
        input_ids,
        assistant_mask,
        pad_mask,
    ):

        with self._autocast():

            with torch.no_grad():
                infer_hidden_states = self.model.forward_backbone(
                    input_ids, mode=ForteMode.INFERENCE
                )
                embeddings = self.model.forward_embeddings(
                    infer_hidden_states,
                    pad_mask,
                )

            hidden_states = self.model.forward_backbone(
                input_ids,
                mode=ForteMode.TRAIN_FIRST,
                embeddings=embeddings,
                embedding_mask=pad_mask,
            )
            lm_states = self.model.forward_lm_states(
                hidden_states,
                mode=ForteMode.TRAIN_FIRST,
                logits_to_keep=slice(0, -1),
                embeddings=embeddings,
                embedding_mask=pad_mask,
            )
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
                mode=ForteMode.TRAIN_SECOND,
            )


    def scanned_second_pass(
        self,
        input_ids,
        assistant_mask,
        pad_mask,
        fast_states,
        fast_grad_buffers,
    ):
        """Pure second-pass body for XLA gradient accumulation."""
        fast_states_leaf = fast_states
        fast_grad_buffers_leaf = fast_grad_buffers

        with self._autocast():
            infer_hidden_states = self.model.forward_backbone(
                input_ids,
                mode=ForteMode.INFERENCE,
                fast_states=fast_states_leaf,
                fast_grad_buffers=fast_grad_buffers_leaf,
            )
            embeddings = self.model.forward_embeddings(
                infer_hidden_states,
                pad_mask,
            )
            hidden_states = self.model.forward_backbone(
                input_ids,
                embeddings,
                pad_mask,
                mode=ForteMode.TRAIN_SECOND,
                future_loss_scale=self.config.trainer.future_loss_scale,
                fast_states=fast_states_leaf,
                fast_grad_buffers=fast_grad_buffers_leaf,
            )
            lm_states = self.model.forward_lm_states(
                hidden_states,
                embeddings,
                pad_mask,
                mode=ForteMode.TRAIN_SECOND,
                logits_to_keep=slice(0, -1),
                future_loss_scale=self.config.trainer.future_loss_scale,
                fast_states=fast_states_leaf,
                fast_grad_buffers=fast_grad_buffers_leaf,
            )
            loss, lm_grad = self.loss_and_lm_grad(
                lm_states,
                input_ids,
                assistant_mask,
            )

        trainable_parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        gradients = torch.autograd.grad(
            lm_states,
            (
                fast_states_leaf,
                fast_grad_buffers_leaf,
                *trainable_parameters,
            ),
            lm_grad,
            allow_unused=True,
            materialize_grads=True,
        )
        state_updates, raw_gradients, *parameter_gradients = gradients
        next_fast_states, next_fast_grad_buffers = (
            self.model.functional_update_state(
                fast_states_leaf,
                fast_grad_buffers_leaf,
                state_updates,
                raw_gradients,
                ForteMode.TRAIN_SECOND,
                state_update_is_scaled=True,
            )
        )

        return (
            loss,
            tuple(parameter_gradients),
            next_fast_states.detach(),
            next_fast_grad_buffers.detach(),
        )


    def scan_second_passes(self, episodes):
        (
            (input_ids, assistant_mask, pad_mask),
            fast_modules,
            fast_states,
            fast_grad_buffers,
        ) = self._scan_tensors(episodes)

        _, fast_states, fast_grad_buffers = self.second_pass_scan(
            (input_ids, assistant_mask, pad_mask),
            fast_states,
            fast_grad_buffers,
        )

        self._store_fast_state(
            fast_modules, fast_states, fast_grad_buffers,
        )


    @torch_xla.compile(full_graph=True)
    def post_forward(self):

        err = self.model.relative_grad_error()
        self.model.empty_state()

        num_none_grad = len([p for p in self.model.parameters() if p.grad is None])

        grad_norm = self.clip_gradients()
        aux = self.optimization_step()

        self.model.zero_grad(set_to_none=False)

        aux["relative_grad_error"] = err
        aux["num_none_grad"] = num_none_grad

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
        assert len(episodes) > 1
        terminal_index = len(episodes) - 1
        losses = []
        aux = {}

        # Carry the fast state through the whole first sweep in one XLA While.
        first_pass_losses = self.scan_first_passes(episodes)
        torch_xla.sync(wait=True)
        for index, loss in enumerate(first_pass_losses.unbind()):
            aux[f"lm_loss/episode_{index:02d}"] = loss
            losses.append(loss)
        master_print(
            f"First passes 00-{terminal_index:02d} completed."
        )

        self.model.finalize_state()
        self.model.zero_grad(set_to_none=False)
        torch_xla.sync(wait=True)

        # Accumulate non-terminal gradients in one sharded XLA While.
        self.scan_second_passes(episodes[:-1])
        torch_xla.sync(wait=True)
        master_print(
            f"Second passes 00-{terminal_index - 1:02d} completed."
        )

        # only do lm loss the last chunk last
        self.terminal_pass(*episodes[-1])
        torch_xla.sync(wait=True)

        master_print(
            f"Second pass {terminal_index:02d} completed."
        )

        # optimizer step
        post_aux, grad_norm = self.post_forward()
        torch_xla.sync(wait=True)

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
