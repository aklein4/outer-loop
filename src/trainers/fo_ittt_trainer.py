import torch
import torch.nn.functional as F
import torch_xla

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
            module.fast_p_r.no_muon = True
            module.fast_p_l.no_muon = True

    def _backward_fast_weight_gradients(self, loss):
        grad_buffers = tuple(
            module.grad_buffer
            for module in self.model._fast_weight_mlps()
        )
        if grad_buffers:
            torch.autograd.backward(
                loss,
                inputs=grad_buffers,
            )

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

    def loss(
        self,
        input_ids: torch.LongTensor,
        assistant_mask: torch.BoolTensor,
        logits: torch.FloatTensor,
    ):
        labels = input_ids[:, 1:].contiguous()
        output_mask = assistant_mask[:, 1:].float().contiguous()

        losses = F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction="none",
        ).view_as(labels)

        output_loss = (
            (losses * output_mask).sum(dim=-1)
            / output_mask.sum(dim=-1).clamp_min(1.0)
        ).mean()

        return output_loss

    @torch_xla.compile(full_graph=True)
    def first_pass(
        self,
        input_ids,
        assistant_mask,
        attention_mask,
    ):
        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            logits, _, hidden_states = self.model(
                input_ids,
                logits_to_keep=slice(0, -1),
                return_states=True,
            )
            loss = self.loss(
                input_ids,
                assistant_mask,
                logits,
            )

        self._backward_fast_weight_gradients(loss)

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

    @torch_xla.compile(full_graph=True)
    def terminal_first_pass(
        self,
        input_ids,
        assistant_mask,
    ):
        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )
        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            logits = self.model(
                input_ids,
                logits_to_keep=slice(0, -1),
            )[0]
            loss = self.loss(
                input_ids,
                assistant_mask,
                logits,
            )

        self._backward_fast_weight_gradients(loss)
        self.model.accumulate_gradients()
        self.model.finalize_gradients()
        self.model.zero_grad(set_to_none=False)
        return loss

    @torch_xla.compile(full_graph=True)
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
            logits = self.model.second_pass_forward(
                double_input_ids,
                embeddings,
                attention_mask,
                logits_to_keep=slice(0, -1),
            )

            loss = self.loss(
                input_ids,
                assistant_mask,
                logits,
            )

        loss.backward()
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
    def terminal_second_pass(
        self,
        input_ids,
        assistant_mask,
    ):
        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )
        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            logits = self.model(
                input_ids,
                logits_to_keep=slice(0, -1),
            )[0]
            loss = self.loss(
                input_ids,
                assistant_mask,
                logits,
            )

        loss.backward()
        self.model.accumulate_gradients()

        relative_grad_error = self.model.relative_grad_error()
        self.model.empty_state()

        grad_norm = self.clip_gradients()
        metrics = self.optimization_step()

        self.model.zero_grad(set_to_none=False)
        self.model.set_fast_weight_mode(
            FastWeightMLP.FIRST_PASS
        )

        metrics["relative_grad_error"] = (
            relative_grad_error
        )
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
        loss = self.terminal_first_pass(
            input_episodes[terminal_index],
            assistant_episodes[terminal_index],
        )
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

        post_metrics, grad_norm = self.terminal_second_pass(
            input_episodes[terminal_index],
            assistant_episodes[terminal_index],
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
