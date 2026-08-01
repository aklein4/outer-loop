import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from models.fo_ittt import FastWeightMLP, FoItttModel
from trainers.base_trainer import BaseTrainer
from utils.logging_utils import master_print
from utils.sharding_utils import maybe_shard_with_gradients


class FoItttTrainer(BaseTrainer):
    model: FoItttModel

    def post_init(self):
        if (
            self.config.trainer.use_autocast
            and "embeddings"
            not in self.config.trainer.multiple_optimizers
        ):
            # The frozen vocabulary projection is already cast to BF16 by
            # autocast for every use. Store it in that compute dtype to avoid
            # retaining both a replicated FP32 weight and its BF16 cast.
            self.model.lm_head.to(dtype=torch.bfloat16)

        self.model.init_state(
            self.global_batch_size,
            self.device,
        )
        for module in self.model._fast_weight_mlps():
            module.fast_log_lr.no_muon = True
            module.fast_p_r.weight.no_muon = True
            module.fast_p_l.weight.no_muon = True
            module.fast_m.no_muon = True

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
                key in name
                for key in ("embed_tokens", "lm_head")
            ):
                embeddings.append(parameter)
            elif any(
                key in name
                for key in (
                    "fast",
                    "embedding_norm",
                    "bidirectional_head",
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
            "embeddings"
            not in self.config.trainer.multiple_optimizers
        ):
            parameters.pop("embeddings")
        return parameters

    def _logit_iteration_loss_and_gradient(
        self,
        lm_states: torch.FloatTensor,
        labels: torch.LongTensor,
        token_weights: torch.FloatTensor,
    ):
        num_iterations = self.config.trainer.num_logit_iterations
        lm_states = lm_states.detach().reshape(
            -1,
            num_iterations,
            lm_states.shape[-1],
        )
        labels = labels.reshape(
            -1,
            num_iterations,
        )
        token_weights = token_weights.reshape(
            -1,
            num_iterations,
        )
        lm_states = maybe_shard_with_gradients(lm_states)
        labels = maybe_shard_with_gradients(labels)
        token_weights = maybe_shard_with_gradients(token_weights)
        lm_states.requires_grad_(True)

        losses = []
        for i in range(num_iterations):
            with self._autocast():
                logits = self.model.lm_head(
                    lm_states[:, i].contiguous()
                ).float()
                loss = (
                    F.cross_entropy(
                        logits,
                        labels[:, i],
                        reduction="none",
                    )
                    * token_weights[:, i]
                ).sum()

            losses.append(loss.detach())
            loss.backward()

            xm.optimization_barrier_([lm_states.grad])

        return (
            torch.stack(losses).sum(),
            lm_states.grad.detach(),
        )

    def loss_and_hidden_gradient(
        self,
        input_ids: torch.LongTensor,
        assistant_mask: torch.BoolTensor,
        hidden_states: torch.FloatTensor,
    ):
        batch_size = hidden_states.shape[0]
        num_iterations = (
            self.config.trainer.num_logit_iterations
        )
        token_count = hidden_states.numel() // hidden_states.shape[-1]
        if token_count % num_iterations != 0:
            raise ValueError(
                "the number of loss tokens must be divisible by "
                "trainer.num_logit_iterations"
            )

        output_mask = assistant_mask[:, 1:].float()
        token_weights = (
            output_mask
            / output_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            / batch_size
        )

        hidden_states_for_grad = maybe_shard_with_gradients(
            hidden_states.detach()
        ).requires_grad_(True)
        with self._autocast():
            lm_states = self.model.model.norm(
                hidden_states_for_grad
            )

        loss, lm_gradient = (
            self._logit_iteration_loss_and_gradient(
                lm_states,
                input_ids[:, 1:],
                token_weights,
            )
        )
        lm_gradient = maybe_shard_with_gradients(
            lm_gradient.reshape_as(lm_states)
        )
        lm_states.backward(lm_gradient)
        if hidden_states_for_grad.grad is None:
            raise RuntimeError(
                "no gradient was produced for the final-norm input"
            )
        hidden_gradient = maybe_shard_with_gradients(
            hidden_states_for_grad.grad.detach().reshape_as(
                hidden_states
            )
        )
        return loss, hidden_gradient

    @torch_xla.compile(full_graph=True)
    def first_pass(
        self,
        input_ids,
        assistant_mask,
        attention_mask,
        update_state=True,
        fast_weight_gradients_only=True,
    ):
        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )

        with self._autocast():
            hidden_states = self.model.backbone_forward(
                input_ids=input_ids,
            )
            loss, hidden_gradient = (
                self.loss_and_hidden_gradient(
                    input_ids,
                    assistant_mask,
                    hidden_states[:, :-1],
                )
            )

        if not fast_weight_gradients_only:
            hidden_states[:, :-1].backward(hidden_gradient)
        elif not self.model.disable_fast_weights:
            torch.autograd.backward(
                hidden_states[:, :-1],
                hidden_gradient,
                inputs=(self.model.fast_weight_grad_buffer,),
            )

        if update_state:
            with torch.no_grad(), self._autocast():
                embeddings = self.model.bidirectional_forward(
                    hidden_states,
                    attention_mask,
                )

            with self._autocast():
                self.model.update_state(
                    embeddings,
                    attention_mask,
                )

        return loss

    @torch_xla.compile(full_graph=True)
    def second_pass(
        self,
        input_ids,
        assistant_mask,
        attention_mask,
    ):
        self.model.set_fast_weight_mode(FastWeightMLP.PLAIN)

        with self._autocast():
            propagated_embeddings = self.model.embedding_forward(
                input_ids,
                attention_mask,
            )

        embeddings = maybe_shard_with_gradients(
            propagated_embeddings.detach()
        ).requires_grad_(True)
        self.model.set_fast_weight_mode(
            FastWeightMLP.SECOND_PASS
        )

        # Interleaving keeps each stream pair on the same batch shard.
        double_input_ids = maybe_shard_with_gradients(
            input_ids[:, None]
            .expand(-1, 2, -1)
            .flatten(0, 1)
        )

        with self._autocast():
            loss_hidden_states = self.model.second_pass_forward(
                double_input_ids,
                embeddings,
                attention_mask,
                logits_to_keep=slice(0, -1),
            )

            _, hidden_gradient = (
                self.loss_and_hidden_gradient(
                    input_ids,
                    assistant_mask,
                    loss_hidden_states,
                )
            )

        loss_hidden_states.backward(hidden_gradient)
        if embeddings.grad is None:
            raise RuntimeError(
                "no gradient was accumulated in current embeddings"
            )
        embedding_gradient = maybe_shard_with_gradients(
            embeddings.grad.detach()
        )

        # The graph that produced the detached learning-rate embeddings is
        # still live. Backpropagating their accumulated gradient avoids a
        # duplicate backbone and bidirectional-head forward.
        self.model.set_fast_weight_mode(FastWeightMLP.PLAIN)
        embedding_loss = (
            propagated_embeddings
            * embedding_gradient.to(propagated_embeddings.dtype)
        ).sum()
        embedding_loss.backward()

        with self._autocast():
            self.model.update_state(
                embeddings,
                attention_mask,
                subtract_gradients=True,
            )

        return embedding_loss

    @torch_xla.compile(full_graph=True)
    def post_forward(self):
        self.model.empty_state()

        grad_norm = self.clip_gradients()
        metrics = self.optimization_step()

        self.model.zero_grad(set_to_none=False)
        return metrics, grad_norm

    def train_step(self, batch):
        input_ids: torch.LongTensor = batch["input_ids"]
        assistant_mask: torch.BoolTensor = batch["assistant_mask"]
        attention_mask: torch.BoolTensor = batch["attention_mask"]

        if input_ids.ndim != 3:
            raise ValueError(
                "input_ids must have shape [batch, horizon, sequence]"
            )
        if (
            assistant_mask.shape != input_ids.shape
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "assistant_mask and attention_mask must match input_ids"
            )
        if input_ids.shape[1] == 0:
            raise ValueError(
                "training horizon must contain at least one episode"
            )

        episodes = tuple(
            zip(
                input_ids.unbind(dim=1),
                assistant_mask.unbind(dim=1),
                attention_mask.unbind(dim=1),
            )
        )
        terminal_index = len(episodes) - 1
        losses = []
        metrics = {}

        for index, episode in enumerate(episodes):
            loss = self.first_pass(
                *episode,
                update_state=index != terminal_index,
            )
            metrics[f"lm_loss/episode_{index:02d}"] = loss
            losses.append(loss)
            master_print(
                f"First-pass horizon {index:02d} completed."
            )

        self.model.accumulate_gradients()
        self.model.finalize_gradients()
        self.model.zero_grad(set_to_none=False)

        for index, episode in enumerate(episodes[:-1]):
            self.second_pass(*episode)
            master_print(
                f"Second-pass horizon {index:02d} completed."
            )

        self.first_pass(
            *episodes[-1],
            update_state=False,
            fast_weight_gradients_only=False,
        )
        self.model.accumulate_gradients(subtract=True)
        relative_grad_error = self.model.relative_grad_error()

        post_metrics, grad_norm = self.post_forward()
        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )
        post_metrics["relative_grad_error"] = relative_grad_error
        master_print(
            f"Second-pass horizon {terminal_index:02d} completed."
        )

        metrics.update(post_metrics)

        final_loss = torch.stack(losses).mean()
        metrics["all_loss"] = final_loss
        metrics["atom_count"] = attention_mask.long().sum()

        decades = {}
        for key, value in metrics.items():
            if "episode_" not in key or key.endswith("00"):
                continue

            decade = int(key.rsplit("_", maxsplit=1)[-1][0])
            decades.setdefault(decade, []).append(
                value
            )

        for decade, values in decades.items():
            metrics[
                f"grouped_lm_loss/decade_{decade:02d}"
            ] = torch.stack(values).mean()

        return final_loss, metrics, grad_norm
