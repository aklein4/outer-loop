import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers.activations import ACT2FN

from models.layers import BidirectionalHead
from models.llama import (
    LlamaForCausalLM,
    LlamaRMSNorm,
)
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import fixed_linear, safe_copy_state


def _raw_fast_weight_gradient(
    activations: torch.FloatTensor,
    output_gradient: torch.FloatTensor,
    down_weight: torch.FloatTensor,
    output_dtype: torch.dtype,
) -> torch.FloatTensor:
    matmul_dtype = torch.bfloat16
    device_type = str(activations.device.type)
    with torch.autocast(
        device_type,
        dtype=matmul_dtype,
    ):
        value_gradient = fixed_linear(
            output_gradient.to(matmul_dtype),
            down_weight.transpose(0, 1).to(matmul_dtype),
        )

    # Match the source implementation: the projection uses the model's BF16
    # precision, but gradients are accumulated over the sequence in FP32.
    with torch.autocast(device_type, enabled=False):
        return (
            value_gradient.transpose(-2, -1).float()
            @ activations.float()
        ).to(output_dtype)


class FastWeightFunction(torch.autograd.Function):
    """Collect raw fast-weight gradients and inject the local FO gradient."""

    @staticmethod
    def forward(
        ctx,
        activations: torch.FloatTensor,
        output: torch.FloatTensor,
        down_weight: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        remaining_gradient: torch.FloatTensor | None,
        learning_rate: torch.FloatTensor | None,
        grad_eps: float,
    ) -> torch.FloatTensor:
        batch_size = grad_buffer.shape[0]
        ctx.second_pass = remaining_gradient is not None
        expected_batch_size = batch_size * (
            2 if ctx.second_pass else 1
        )
        if activations.shape[0] != expected_batch_size:
            raise ValueError(
                f"expected fast-weight batch {expected_batch_size}, "
                f"got {activations.shape[0]}"
            )

        if ctx.second_pass:
            if learning_rate is None:
                raise ValueError(
                    "learning_rate is required during the second pass"
                )
            ctx.save_for_backward(
                activations.reshape(
                    batch_size,
                    2,
                    *activations.shape[1:],
                )[:, 0],
                down_weight,
                remaining_gradient,
                learning_rate,
            )
            ctx.activation_dtype = activations.dtype
        else:
            ctx.save_for_backward(activations, down_weight)

        ctx.grad_dtype = grad_buffer.dtype
        ctx.grad_eps = grad_eps
        return output

    @staticmethod
    def backward(
        ctx,
        output_gradient: torch.FloatTensor,
    ):
        if not ctx.second_pass:
            activations, down_weight = ctx.saved_tensors
            raw_gradient = _raw_fast_weight_gradient(
                activations,
                output_gradient,
                down_weight,
                ctx.grad_dtype,
            )

            return (
                None,
                output_gradient,
                None,
                raw_gradient,
                None,
                None,
                None,
            )

        (
            activations,
            down_weight,
            remaining_gradient,
            learning_rate,
        ) = ctx.saved_tensors

        batch_size = remaining_gradient.shape[0]
        lm_output_gradient = output_gradient.reshape(
            batch_size,
            2,
            *output_gradient.shape[1:],
        )[:, 0].detach()

        with torch.enable_grad():
            activations_for_grad = (
                activations.detach()
                .requires_grad_(True)
            )
            down_weight_for_grad = (
                down_weight.detach()
                .requires_grad_(True)
            )
            learning_rate_for_grad = (
                learning_rate.detach()
                .float()
                .requires_grad_(True)
            )

            local_raw_gradient = _raw_fast_weight_gradient(
                activations_for_grad,
                lm_output_gradient,
                down_weight_for_grad,
                ctx.grad_dtype,
            )
            raw_gradient = local_raw_gradient.detach()
            future_gradient = (
                remaining_gradient.to(ctx.grad_dtype)
                - raw_gradient
            ).detach()

            normalized_gradient = F.rms_norm(
                local_raw_gradient,
                local_raw_gradient.shape[-2:],
                eps=ctx.grad_eps,
            )
            state_update = -(
                learning_rate_for_grad * normalized_gradient
            )
            local_loss = (future_gradient * state_update).sum()
            (
                activation_gradient,
                down_weight_gradient,
                learning_rate_gradient,
            ) = torch.autograd.grad(
                local_loss,
                (
                    activations_for_grad,
                    down_weight_for_grad,
                    learning_rate_for_grad,
                ),
            )

        activation_gradient = torch.stack(
            (
                torch.zeros_like(activation_gradient),
                activation_gradient,
            ),
            dim=1,
        ).flatten(0, 1).to(ctx.activation_dtype)
        activation_gradient = maybe_shard_with_gradients(
            activation_gradient
        )

        return (
            activation_gradient,
            output_gradient,
            down_weight_gradient.to(down_weight.dtype),
            raw_gradient,
            None,
            learning_rate_gradient.to(learning_rate.dtype),
            None,
        )


class FastWeightMLP(nn.Module):
    FIRST_PASS = "first"
    SECOND_PASS = "second"
    PLAIN = "plain"

    def __init__(
        self,
        config: DictConfig,
        layer_index: int,
    ):
        super().__init__()

        self.intermediate_size = config.intermediate_size
        self.fast_weight_size = config.fast_weight_size

        self.base_lr = config.base_lr
        self.grad_eps = config.grad_rms_eps
        self.state_dtype = getattr(torch, config.state_dtype)
        self.scalar_scaler = math.sqrt(config.hidden_size)

        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.up_fast = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=True,
        )
        self.gate_fast = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=True,
        )
        self.down_fast = nn.Linear(
            self.fast_weight_size,
            config.hidden_size,
            bias=False,
        )

        self.fast_log_lr = nn.Parameter(
            torch.empty(
                self.fast_weight_size,
                self.fast_weight_size,
            )
        )

        self.fast_p_r = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=False,
        )
        self.fast_p_l = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=False,
        )
        for parameter in (
            self.fast_log_lr,
            self.fast_p_r.weight,
            self.fast_p_l.weight,
        ):
            parameter.no_muon = True
        self.register_buffer(
            "fast_weight_index",
            torch.tensor(layer_index, dtype=torch.int32),
            persistent=False,
        )

        self.mode = self.FIRST_PASS

    @torch.no_grad()
    def reset_fast_parameters(self, initializer_range: float):
        for projection in (self.up_fast, self.gate_fast):
            projection.weight.normal_(std=initializer_range)
            projection.bias.zero_()
        self.down_fast.weight.normal_(std=initializer_range)

        self.fast_log_lr.normal_(std=0.25 / self.scalar_scaler)
        projection_std = 0.5 / self.scalar_scaler
        self.fast_p_r.weight.normal_(std=projection_std)
        self.fast_p_l.weight.normal_(std=projection_std)

    def get_lr(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        embedding_mask = embedding_mask.to(embeddings.dtype)
        masked_embeddings = embeddings * embedding_mask[..., None]
        count = embedding_mask.sum(dim=-1).clamp_min(1.0)

        offset = (
            self.fast_p_l(masked_embeddings).transpose(-2, -1)
            @ self.fast_p_r(masked_embeddings)
        ) / count[..., None, None]
        offset = -F.elu(-offset)

        return (
            self.base_lr
            * torch.exp(
                self.fast_log_lr * self.scalar_scaler
                + offset
            )
            / math.sqrt(self.fast_weight_size)
        )

    def set_mode(self, mode: str):
        if mode not in (
            self.FIRST_PASS,
            self.SECOND_PASS,
            self.PLAIN,
        ):
            raise ValueError(f"unknown fast-weight mode: {mode}")
        self.mode = mode

    def _select_layer(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # XLA scan consumes the leading layer axis before calling the layer.
        if tensor.ndim == 3:
            return tensor
        return torch.index_select(
            tensor,
            0,
            self.fast_weight_index[None],
        ).squeeze(0)

    def forward(
        self,
        x: torch.FloatTensor,
        fast_weight_state: torch.FloatTensor,
        fast_weight_grad_buffer: torch.FloatTensor,
        fast_weight_embeddings: torch.FloatTensor | None = None,
        fast_weight_embedding_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        base_output = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )
        fast_hidden = (
            self.up_fast(x) * F.silu(self.gate_fast(x))
        )

        state = self._select_layer(fast_weight_state).detach()
        if self.mode != self.PLAIN:
            grad_buffer = self._select_layer(
                fast_weight_grad_buffer
            )
        if self.mode == self.SECOND_PASS:
            batch_size = state.shape[0]
            # Streams are interleaved per example. The reshape therefore keeps
            # the original batch sharding on `batch_size` and broadcasts the
            # state over a local size-two stream axis.
            fast_hidden_streams = fast_hidden.reshape(
                batch_size,
                2,
                *fast_hidden.shape[1:],
            )
            fast_values = torch.einsum(
                "boi,bnsi->bnso",
                state,
                fast_hidden_streams,
            ).flatten(0, 1)
            fast_values = maybe_shard_with_gradients(
                fast_values
            )
        else:
            fast_values = torch.einsum(
                "boi,bsi->bso",
                state,
                fast_hidden,
            )
        fast_output = self.down_fast(fast_values)

        remaining_gradient = learning_rate = None
        if self.mode == self.SECOND_PASS:
            if (
                fast_weight_embeddings is None
                or fast_weight_embedding_mask is None
            ):
                raise RuntimeError(
                    "second pass requires current learning-rate embeddings"
                )

            learning_rate = self.get_lr(
                fast_weight_embeddings,
                fast_weight_embedding_mask,
            )
            remaining_gradient = grad_buffer.detach()

        if self.mode != self.PLAIN:
            fast_output = FastWeightFunction.apply(
                fast_hidden,
                fast_output,
                self.down_fast.weight,
                grad_buffer,
                remaining_gradient,
                learning_rate,
                self.grad_eps,
            )

        return base_output + fast_output

    @torch.no_grad()
    def accumulate_gradients(
        self,
        grad_buffer: torch.FloatTensor,
        raw_gradient: torch.FloatTensor,
        subtract: bool = False,
    ):
        gradient = raw_gradient.to(grad_buffer.dtype)
        if subtract:
            grad_buffer.sub_(gradient)
        else:
            grad_buffer.add_(gradient)
        raw_gradient.zero_()

    @torch.no_grad()
    def update_state(
        self,
        state: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        raw_gradient: torch.FloatTensor,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.FloatTensor,
        subtract_gradients: bool = False,
    ):
        normalized_gradient = F.rms_norm(
            raw_gradient.float(),
            raw_gradient.shape[-2:],
            eps=self.grad_eps,
        )
        learning_rate = self.get_lr(
            embeddings,
            embedding_mask,
        )
        update = -(
            learning_rate.to(normalized_gradient.dtype)
            * normalized_gradient
        )

        state.add_(update.to(state.dtype))
        self.accumulate_gradients(
            grad_buffer,
            raw_gradient,
            subtract=subtract_gradients,
        )

    @torch.no_grad()
    def finalize_gradients(
        self,
        state: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        raw_gradient: torch.FloatTensor,
        final_grad_norm: torch.FloatTensor,
    ):
        state.zero_()
        # Keep the first-pass total in grad_buffer as a second-pass countdown.
        # Only its norm must survive after that countdown has been consumed.
        final_grad_norm.copy_(grad_buffer.float().norm())
        raw_gradient.zero_()

    @torch.no_grad()
    def empty_state(
        self,
        state: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        raw_gradient: torch.FloatTensor,
        final_grad_norm: torch.FloatTensor,
    ):
        state.zero_()
        grad_buffer.zero_()
        raw_gradient.zero_()
        final_grad_norm.zero_()

    @torch.no_grad()
    def relative_grad_error(
        self,
        grad_buffer: torch.FloatTensor,
        final_grad_norm: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return (
            grad_buffer.norm()
            / (final_grad_norm + self.grad_eps)
        )

    @torch.no_grad()
    def copy_fast_weights_from_base(self):
        if self.fast_weight_size > self.intermediate_size:
            raise ValueError(
                "fast_weight_size must not exceed intermediate_size"
            )
        size = self.fast_weight_size
        self.up_fast.weight.copy_(self.up_proj.weight[:size])
        self.gate_fast.weight.copy_(self.gate_proj.weight[:size])
        self.down_fast.weight.copy_(self.down_proj.weight[:, :size])


class FoItttModel(LlamaForCausalLM):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.embedding_norm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.bidirectional_head = BidirectionalHead(config)
        self.bidirectional_head.apply(self._init_weights)

        self.disable_fast_weights = config.get(
            "disable_fast_weights",
            False,
        )
        if not self.disable_fast_weights:
            for index, layer in enumerate(self.model.layers):
                fast_mlp = FastWeightMLP(config, index)
                fast_mlp.apply(self._init_weights)
                fast_mlp.reset_fast_parameters(
                    config.initializer_range
                )
                safe_copy_state(
                    layer.mlp,
                    fast_mlp,
                    strict=False,
                )
                layer.mlp = fast_mlp

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        fast_weight_suffixes = (
            ".up_fast.weight",
            ".gate_fast.weight",
            ".down_fast.weight",
        )
        has_fast_weights = any(
            key.endswith(fast_weight_suffixes)
            for key in state_dict
        )

        if self.disable_fast_weights or has_fast_weights:
            return super().load_state_dict(
                state_dict,
                strict=strict,
                assign=assign,
            )

        incompatible_keys = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )

        for mlp in self._fast_weight_mlps():
            mlp.copy_fast_weights_from_base()

        return incompatible_keys

    def _layer_mlp(self, layer) -> FastWeightMLP:
        try:
            return layer.get_submodule("mlp")
        except AttributeError:
            return layer._orig_mod.get_submodule("mlp")

    def _fast_weight_mlps(self) -> list[FastWeightMLP]:
        if self.disable_fast_weights:
            return []
        return [
            self._layer_mlp(layer) for layer in self.model.layers
        ]

    def set_fast_weight_mode(self, mode: str):
        for module in self._fast_weight_mlps():
            module.set_mode(mode)

    def _prepare_backbone_kwargs(self, kwargs):
        if self.disable_fast_weights:
            kwargs.pop("fast_weight_embeddings", None)
            kwargs.pop("fast_weight_embedding_mask", None)
            return kwargs

        embeddings = kwargs.get("fast_weight_embeddings")
        embedding_mask = kwargs.get(
            "fast_weight_embedding_mask"
        )
        if (embeddings is None) != (embedding_mask is None):
            raise ValueError(
                "fast-weight embeddings and mask must be provided together"
            )
        if embedding_mask is not None:
            # Convert before entering scan so all layer arguments are floating
            # tensors during functional-call/FakeTensor tracing.
            kwargs["fast_weight_embedding_mask"] = embedding_mask.float()

        layer_count = len(self.model.layers)
        if any(
            tensor.shape[0] != layer_count
            for tensor in (
                self.fast_weight_state,
                self.fast_weight_grad_buffer,
            )
        ):
            raise ValueError(
                "fast-weight state and gradient buffer must have one "
                "entry per layer"
            )
        runtime_tensors = dict(
            fast_weight_state=self.fast_weight_state,
            fast_weight_grad_buffer=self.fast_weight_grad_buffer,
        )
        if getattr(
            self.model.layers,
            "supports_scan_inputs",
            False,
        ):
            kwargs["_scan_inputs"] = runtime_tensors
        else:
            kwargs.update(runtime_tensors)
        return kwargs

    def backbone_forward(self, *args, **kwargs):
        return self.model(
            *args,
            **self._prepare_backbone_kwargs(kwargs),
        )

    def forward(self, *args, **kwargs):
        return super().forward(
            *args,
            **self._prepare_backbone_kwargs(kwargs),
        )

    def bidirectional_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        hidden_states = self.embedding_norm(hidden_states)
        return self.bidirectional_head(
            hidden_states,
            elementwise_pad_mask=attention_mask,
        )

    def embedding_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        hidden_states = self.backbone_forward(
            input_ids=input_ids
        )
        return self.bidirectional_forward(hidden_states, attention_mask)

    def second_pass_forward(
        self,
        input_ids: torch.LongTensor,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.FloatTensor,
        logits_to_keep: slice,
    ) -> torch.FloatTensor:
        """Run both residual streams and return the loss stream."""
        hidden_states = self.backbone_forward(
            input_ids=input_ids,
            fast_weight_embeddings=embeddings,
            fast_weight_embedding_mask=embedding_mask,
        )
        hidden_states = hidden_states.reshape(
            embeddings.shape[0],
            2,
            *hidden_states.shape[1:],
        )[:, 0, logits_to_keep, :].contiguous()
        return maybe_shard_with_gradients(hidden_states)

    @torch.no_grad()
    def init_state(self, batch_size: int, device: torch.device):
        modules = self._fast_weight_mlps()
        if not modules:
            return

        fast_weight_size = modules[0].fast_weight_size
        state = torch.zeros(
            len(modules),
            batch_size,
            fast_weight_size,
            fast_weight_size,
            device=device,
            dtype=modules[0].state_dtype,
        )
        grad_buffer = torch.zeros_like(
            state,
            dtype=torch.float32,
        )
        final_grad_norm = torch.zeros(
            len(modules),
            device=device,
            dtype=torch.float32,
        )

        # Layer is the scan dimension and must remain replicated. Batch keeps
        # the same data/FSDP sharding each per-layer buffer previously used.
        state_spec = (
            None,
            ("data", "fsdp"),
            None,
            None,
        )
        state = maybe_shard_with_gradients(
            state,
            spec=state_spec,
        )
        grad_buffer = maybe_shard_with_gradients(
            grad_buffer,
            spec=state_spec,
        )
        grad_buffer.requires_grad_(True)
        grad_buffer.grad = maybe_shard_with_gradients(
            torch.zeros_like(grad_buffer),
            spec=state_spec,
        )

        self.register_buffer(
            "fast_weight_state",
            state,
            persistent=False,
        )
        self.register_buffer(
            "fast_weight_grad_buffer",
            grad_buffer,
            persistent=False,
        )
        self.register_buffer(
            "fast_weight_final_grad_norm",
            final_grad_norm,
            persistent=False,
        )

    def _iter_fast_weight_runtime(self):
        """Yield each fast MLP with its model-owned runtime tensors."""
        modules = self._fast_weight_mlps()
        if not modules:
            return

        raw_gradient = self.fast_weight_grad_buffer.grad
        if raw_gradient is None:
            raise RuntimeError(
                "fast-weight gradient buffer has no gradient storage"
            )
        for index, module in enumerate(modules):
            yield (
                module,
                self.fast_weight_state[index],
                self.fast_weight_grad_buffer[index],
                raw_gradient[index],
                self.fast_weight_final_grad_norm[index],
            )

    @torch.no_grad()
    def accumulate_gradients(self, subtract: bool = False):
        for module, _, grad_buffer, raw_gradient, _ in (
            self._iter_fast_weight_runtime()
        ):
            module.accumulate_gradients(
                grad_buffer,
                raw_gradient,
                subtract=subtract,
            )

    @torch.no_grad()
    def update_state(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
        subtract_gradients: bool = False,
    ):
        for module, state, grad_buffer, raw_gradient, _ in (
            self._iter_fast_weight_runtime()
        ):
            module.update_state(
                state,
                grad_buffer,
                raw_gradient,
                embeddings,
                embedding_mask,
                subtract_gradients=subtract_gradients,
            )

    @torch.no_grad()
    def finalize_gradients(self):
        for module, state, buffer, gradient, norm in (
            self._iter_fast_weight_runtime()
        ):
            module.finalize_gradients(
                state,
                buffer,
                gradient,
                norm,
            )

    @torch.no_grad()
    def empty_state(self):
        for module, state, buffer, gradient, norm in (
            self._iter_fast_weight_runtime()
        ):
            module.empty_state(
                state,
                buffer,
                gradient,
                norm,
            )

    @torch.no_grad()
    def relative_grad_error(self) -> torch.FloatTensor:
        errors = []
        for module, _, grad_buffer, _, norm in (
            self._iter_fast_weight_runtime()
        ):
            errors.append(
                module.relative_grad_error(grad_buffer, norm)
            )

        if not errors:
            return torch.zeros(
                (),
                device=self.model.embed_tokens.weight.device,
            )
        return torch.stack(errors).mean()
