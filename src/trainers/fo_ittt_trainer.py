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

            with torch.no_grad():
                embeddings = self.model.bidirectional_forward(
                    hidden_states,
                    attention_mask,
                )

        loss.backward()

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
    def begin_second_pass(self):
        self.model.finalize_gradients()
        self.model.zero_grad(set_to_none=False)

    @torch_xla.compile(full_graph=True)
    def second_pass(
        self,
        input_ids,
        assistant_mask,
        attention_mask,
    ):
        self.model.set_fast_weight_mode(FastWeightMLP.PLAIN)

        with torch.no_grad():
            with torch.autocast(
                "xla",
                dtype=torch.bfloat16,
                enabled=self.config.trainer.use_autocast,
            ):
                embeddings = self.model.embedding_forward(
                    input_ids,
                    attention_mask,
                )

        embeddings = maybe_shard_with_gradients(
            embeddings.detach()
        )
        embeddings.requires_grad_(True)
        self.model.set_fast_weight_mode(
            FastWeightMLP.SECOND_PASS
        )

        double_input_ids = maybe_shard_with_gradients(
            input_ids.repeat(2, 1)
        )

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            logits = self.model(
                double_input_ids,
                logits_to_keep=slice(0, -1),
                fast_weight_embeddings=embeddings,
                fast_weight_embedding_mask=attention_mask,
            )[0]
            logits = maybe_shard_with_gradients(
                logits[:input_ids.shape[0]]
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
        embedding_gradient = embeddings.grad.detach()
        embedding_gradient = maybe_shard_with_gradients(
            embedding_gradient
        )

        # Recompute the same pre-update embedding graph. This propagates the
        # adaptive-learning-rate gradient through both the bidirectional head
        # and the entire causal backbone.
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

        horizon_length = input_ids.shape[1]
        self.model.empty_state()

        total_loss = 0.0
        metrics = {}

        for index in range(horizon_length):
            loss = self.first_pass(
                input_ids[:, index],
                assistant_mask[:, index],
                attention_mask[:, index],
            )
            torch_xla.sync(wait=True)

            metrics[
                f"lm_loss/episode_{index:02d}"
            ] = loss
            total_loss = total_loss + loss

            master_print(
                f"First-pass horizon {index:02d} completed."
            )

        self.begin_second_pass()
        torch_xla.sync(wait=True)

        for index in range(horizon_length):
            self.second_pass(
                input_ids[:, index],
                assistant_mask[:, index],
                attention_mask[:, index],
            )
            torch_xla.sync(wait=True)
            master_print(
                f"Second-pass horizon {index:02d} completed."
            )

        post_metrics, grad_norm = self.post_forward()
        torch_xla.sync(wait=True)
        metrics.update(post_metrics)

        final_loss = total_loss / horizon_length
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
