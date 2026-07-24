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
        self.model.init_state(
            self.global_batch_size,
            self.device,
        )
        for module in self.model._fast_weight_mlps():
            module.fast_log_lr.no_muon = True
            module.fast_p_r.weight.no_muon = True
            module.fast_p_l.weight.no_muon = True

    def get_trainable_parameters(self, model):
        slow = []
        fast = []
        embeddings = []

        for name, parameter in model.named_parameters():
            if (
                "embed_tokens" in name
                or "lm_head" in name
            ):
                embeddings.append(parameter)
            elif (
                "fast" in name
                or "embedding_norm" in name
                or "bidirectional_head" in name
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
        hidden_states: torch.FloatTensor,
        labels: torch.LongTensor,
        token_weights: torch.FloatTensor,
    ):
        num_iterations = self.config.trainer.num_logit_iterations
        hidden_states = hidden_states.detach().reshape(
            -1,
            num_iterations,
            hidden_states.shape[-1],
        )
        labels = labels.reshape(
            -1,
            num_iterations,
        )
        token_weights = token_weights.reshape(
            -1,
            num_iterations,
        )
        hidden_states = maybe_shard_with_gradients(hidden_states)
        labels = maybe_shard_with_gradients(labels)
        token_weights = maybe_shard_with_gradients(token_weights)
        hidden_states.requires_grad_(True)

        losses = []
        for i in range(num_iterations):
            hs = hidden_states[:, i].contiguous()

            with torch.autocast(
                "xla",
                dtype=torch.bfloat16,
                enabled=self.config.trainer.use_autocast,
            ):
                lm_states = self.model.model.norm(hs)
                logits = self.model.lm_head(lm_states).float()
                token_losses = F.cross_entropy(
                    logits,
                    labels[:, i],
                    reduction="none",
                )
                loss = (
                    token_losses
                    * token_weights[:, i]
                ).sum()

            xm.optimization_barrier_([loss])

            losses.append(loss.detach())
            loss.backward()

        return (
            torch.stack(losses).sum(),
            hidden_states.grad.detach(),
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
        iteration_size = token_count // num_iterations
        if iteration_size % self.logit_batch_shards != 0:
            raise ValueError(
                "each logit iteration must be divisible by the "
                "number of data/FSDP shards"
            )

        output_mask = assistant_mask[:, 1:].float()
        token_weights = (
            output_mask
            / output_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            / batch_size
        )

        loss, hidden_gradient = (
            self._logit_iteration_loss_and_gradient(
                hidden_states,
                input_ids[:, 1:],
                token_weights,
            )
        )
        hidden_gradient = maybe_shard_with_gradients(
            hidden_gradient.reshape_as(hidden_states)
        )
        return loss, hidden_gradient

    def first_pass(
        self,
        input_ids,
        assistant_mask,
        attention_mask,
        update_state=True,
    ):
        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            hidden_states = self.model.model(
                input_ids=input_ids,
            )
            loss, hidden_gradient = (
                self.loss_and_hidden_gradient(
                    input_ids,
                    assistant_mask,
                    hidden_states[:, :-1],
                )
            )

        grad_buffers = tuple(
            module.grad_buffer
            for module in self.model._fast_weight_mlps()
        )
        if grad_buffers:
            torch.autograd.backward(
                hidden_states[:, :-1],
                hidden_gradient,
                inputs=grad_buffers,
            )

        if update_state:
            with torch.no_grad():
                with torch.autocast(
                    "xla",
                    dtype=torch.bfloat16,
                    enabled=self.config.trainer.use_autocast,
                ):
                    embeddings = self.model.bidirectional_forward(
                        hidden_states,
                        attention_mask,
                    )

            with torch.autocast(
                "xla",
                dtype=torch.bfloat16,
                enabled=self.config.trainer.use_autocast,
            ):
                self.model.update_state(
                    embeddings,
                    attention_mask,
                )

        return loss

    def second_pass(
        self,
        input_ids,
        assistant_mask,
        attention_mask,
    ):
        self.model.set_fast_weight_mode(FastWeightMLP.PLAIN)

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            propagated_embeddings = self.model.embedding_forward(
                input_ids,
                attention_mask,
            )

        embeddings = maybe_shard_with_gradients(
            propagated_embeddings.detach()
        )
        embeddings.requires_grad_(True)
        self.model.set_fast_weight_mode(
            FastWeightMLP.SECOND_PASS
        )

        # Interleaving keeps each stream pair on the same batch shard.
        double_input_ids = maybe_shard_with_gradients(
            input_ids[:, None]
            .expand(-1, 2, -1)
            .flatten(0, 1)
        )

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
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

        torch.autograd.backward(
            loss_hidden_states,
            hidden_gradient,
        )
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
            * embedding_gradient.to(
                propagated_embeddings.dtype
            )
        ).sum()
        embedding_loss.backward()

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            self.model.update_state(
                embeddings,
                attention_mask,
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
        assistant_mask: torch.BoolTensor = batch[
            "assistant_mask"
        ]
        attention_mask: torch.BoolTensor = batch[
            "attention_mask"
        ]

        if input_ids.ndim != 3:
            raise ValueError(
                "input_ids must have shape "
                "[batch, horizon, sequence]"
            )
        if (
            assistant_mask.shape != input_ids.shape
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "assistant_mask and attention_mask must match "
                "input_ids"
            )

        input_episodes = input_ids.unbind(dim=1)
        assistant_episodes = assistant_mask.unbind(dim=1)
        attention_episodes = attention_mask.unbind(dim=1)
        horizon_length = len(input_episodes)

        losses = []
        metrics = {}

        for index in range(horizon_length - 1):
            loss = self.first_pass(
                input_episodes[index],
                assistant_episodes[index],
                attention_episodes[index],
            )

            metrics[
                f"lm_loss/episode_{index:02d}"
            ] = loss
            losses.append(loss)

            master_print(
                f"First-pass horizon {index:02d} completed."
            )

        terminal_index = horizon_length - 1
        loss = self.first_pass(
            input_episodes[terminal_index],
            assistant_episodes[terminal_index],
            attention_episodes[terminal_index],
            update_state=False,
        )
        self.model.accumulate_gradients()
        self.model.finalize_gradients()
        self.model.zero_grad(set_to_none=False)

        metrics[
            f"lm_loss/episode_{terminal_index:02d}"
        ] = loss
        losses.append(loss)
        master_print(
            f"First-pass horizon {terminal_index:02d} completed."
        )

        for index in range(horizon_length - 1):
            self.second_pass(
                input_episodes[index],
                assistant_episodes[index],
                attention_episodes[index],
            )
            master_print(
                f"Second-pass horizon {index:02d} completed."
            )

        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )
        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            hidden_states = self.model.model(
                input_ids=input_episodes[terminal_index],
            )
            _, hidden_gradient = (
                self.loss_and_hidden_gradient(
                    input_episodes[terminal_index],
                    assistant_episodes[terminal_index],
                    hidden_states[:, :-1],
                )
            )

        torch.autograd.backward(
            hidden_states[:, :-1],
            hidden_gradient,
        )
        self.model.accumulate_gradients()
        relative_grad_error = self.model.relative_grad_error()

        post_metrics, grad_norm = self.post_forward()
        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )
        post_metrics["relative_grad_error"] = (
            relative_grad_error
        )
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
