import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers.activations import ACT2FN

from models.layers import BidirectionalHead
from models.llama import (
    LlamaDecoderLayer,
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

        if activations.shape[0] == batch_size:
            ctx.second_pass = False
            ctx.save_for_backward(activations, down_weight)
        elif activations.shape[0] == 2 * batch_size:
            if (
                remaining_gradient is None
                or learning_rate is None
            ):
                raise ValueError(
                    "remaining_gradient and learning_rate are required "
                    "during the second pass"
                )

            ctx.second_pass = True
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
            raise ValueError(
                "fast-weight batch must equal the state batch on the first "
                "pass or twice the state batch on the second pass"
            )

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


class UnitGLU(nn.Module):
    def forward(self, x, gate):
        return x * F.silu(gate) / 0.6

class StandardGLU(nn.Module):
    def forward(self, x, gate):
        return x * F.silu(gate)


class FastWeightMLP(nn.Module):
    FIRST_PASS = "first"
    SECOND_PASS = "second"
    PLAIN = "plain"

    def __init__(self, config: DictConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fast_weight_size = config.fast_weight_size

        self.base_lr = config.base_lr
        self.grad_eps = config.grad_rms_eps
        self.state_dtype = getattr(torch, config.state_dtype)
        self.scalar_scaler = math.sqrt(self.hidden_size)

        self.act_fn = ACT2FN[config.hidden_act]
        self.fast_act_fn = StandardGLU()

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )

        self.up_fast = nn.Linear(
            self.hidden_size,
            self.fast_weight_size,
            bias=True,
        )
        self.gate_fast = nn.Linear(
            self.hidden_size,
            self.fast_weight_size,
            bias=True,
        )
        self.down_fast = nn.Linear(
            self.fast_weight_size,
            self.hidden_size,
            bias=False,
        )

        self.fast_log_lr = nn.Parameter(
            torch.empty(
                self.fast_weight_size,
                self.fast_weight_size,
            )
        )
        self.fast_log_lr.no_muon = True

        self.fast_p_r = nn.Linear(
            self.hidden_size,
            self.fast_weight_size,
            bias=False,
        )
        self.fast_p_l = nn.Linear(
            self.hidden_size,
            self.fast_weight_size,
            bias=False,
        )

        self.state: nn.Buffer
        self.grad_buffer: nn.Buffer
        self.final_grad_buffer: nn.Buffer

        self.mode = self.FIRST_PASS

    @torch.no_grad()
    def reset_fast_parameters(self, initializer_range: float):
        self.up_fast.weight.normal_(std=initializer_range)
        self.up_fast.bias.zero_()
        self.gate_fast.weight.normal_(std=initializer_range)
        self.gate_fast.bias.zero_()
        self.down_fast.weight.normal_(std=initializer_range)

        self.fast_log_lr.normal_(
            std=0.25 / self.scalar_scaler
        )
        projection_std = 0.5 / math.sqrt(self.hidden_size)
        self.fast_p_r.weight.normal_(std=projection_std)
        self.fast_p_l.weight.normal_(std=projection_std)

    def get_lr(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        masked_embeddings = (
            embeddings
            * embedding_mask[..., None].to(embeddings.dtype)
        )
        count = (
            embedding_mask.to(embeddings.dtype)
            .sum(dim=-1)
            .clamp_min(1.0)
        )

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
        if mode not in {
            self.FIRST_PASS,
            self.SECOND_PASS,
            self.PLAIN,
        }:
            raise ValueError(f"unknown fast-weight mode: {mode}")
        self.mode = mode

    def forward(
        self,
        x: torch.FloatTensor,
        fast_weight_embeddings: torch.FloatTensor | None = None,
        fast_weight_embedding_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        base_hidden = (
            self.act_fn(self.gate_proj(x))
            * self.up_proj(x)
        )
        base_output = self.down_proj(base_hidden)

        fast_hidden = self.fast_act_fn(
            self.up_fast(x),
            self.gate_fast(x),
        )

        state = self.state.detach()
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

        if self.mode == self.FIRST_PASS:
            fast_output = FastWeightFunction.apply(
                fast_hidden,
                fast_output,
                self.down_fast.weight,
                self.grad_buffer,
                None,
                None,
                self.grad_eps,
            )
        elif self.mode == self.SECOND_PASS:
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
            remaining_gradient = (
                self.final_grad_buffer
                - self.grad_buffer.to(
                    self.final_grad_buffer.dtype
                )
            ).detach()
            fast_output = FastWeightFunction.apply(
                fast_hidden,
                fast_output,
                self.down_fast.weight,
                self.grad_buffer,
                remaining_gradient,
                learning_rate,
                self.grad_eps,
            )

        return base_output + fast_output

    @torch.no_grad()
    def init_state(self, batch_size: int, device: torch.device):
        state = torch.zeros(
            batch_size,
            self.fast_weight_size,
            self.fast_weight_size,
            device=device,
            dtype=self.state_dtype,
        )
        grad_buffer = torch.zeros_like(
            state,
            dtype=torch.float32,
        )
        final_grad_buffer = torch.zeros_like(
            state,
            dtype=torch.float32,
        )

        state = maybe_shard_with_gradients(state)
        grad_buffer = maybe_shard_with_gradients(grad_buffer)
        final_grad_buffer = maybe_shard_with_gradients(
            final_grad_buffer
        )

        self.register_buffer("state", state, persistent=False)
        self.register_buffer(
            "grad_buffer",
            grad_buffer,
            persistent=False,
        )
        self.register_buffer(
            "final_grad_buffer",
            final_grad_buffer,
            persistent=False,
        )

        self.state.requires_grad_(False)
        self.grad_buffer.requires_grad_(True)
        self.grad_buffer.grad = maybe_shard_with_gradients(
            torch.zeros_like(self.grad_buffer)
        )
        self.final_grad_buffer.requires_grad_(False)

    @torch.no_grad()
    def accumulate_gradients(self):
        self.grad_buffer.add_(
            self.grad_buffer.grad.to(self.grad_buffer.dtype)
        )
        self.grad_buffer.grad.zero_()

    @torch.no_grad()
    def update_state(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.FloatTensor,
    ):
        raw_gradient = self.grad_buffer.grad
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

        self.state.add_(update.to(self.state.dtype))
        self.grad_buffer.add_(
            raw_gradient.to(self.grad_buffer.dtype)
        )
        self.grad_buffer.grad.zero_()

    @torch.no_grad()
    def finalize_gradients(self):
        self.state.zero_()
        self.final_grad_buffer.copy_(
            self.grad_buffer.to(self.final_grad_buffer.dtype)
        )
        self.grad_buffer.zero_()
        self.grad_buffer.grad.zero_()

    @torch.no_grad()
    def empty_state(self):
        self.state.zero_()
        self.grad_buffer.zero_()
        self.grad_buffer.grad.zero_()
        self.final_grad_buffer.zero_()

    @torch.no_grad()
    def relative_grad_error(self) -> torch.FloatTensor:
        error = (
            self.grad_buffer
            - self.final_grad_buffer.to(self.grad_buffer.dtype)
        ).norm()
        denominator = (
            self.final_grad_buffer.norm()
            + self.grad_eps
        )
        return error / denominator


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
            for layer in self.model.layers:
                layer: LlamaDecoderLayer

                fast_mlp = FastWeightMLP(config)
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
        # TODO: port this to oloop
        
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

        super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )

        with torch.no_grad():
            for mlp in self._fast_weight_mlps():

                fast_weight_size = mlp.fast_weight_size
                if fast_weight_size > mlp.intermediate_size:
                    raise ValueError(
                        "fast_weight_size must not exceed "
                        "intermediate_size when initializing fast weights "
                        "from regular MLP weights"
                    )

                mlp.up_fast.weight.copy_(
                    mlp.up_proj.weight[:fast_weight_size]
                )
                mlp.gate_fast.weight.copy_(
                    mlp.gate_proj.weight[:fast_weight_size]
                )
                mlp.down_fast.weight.copy_(
                    mlp.down_proj.weight[:, :fast_weight_size]
                )


    def _layer_mlp(self, layer) -> FastWeightMLP:
        try:
            return layer.get_submodule("mlp")
        except AttributeError:
            return layer._orig_mod.get_submodule("mlp")

    def _fast_weight_mlps(self) -> list[FastWeightMLP]:
        if self.disable_fast_weights:
            return []
        return [
            self._layer_mlp(layer)
            for layer in self.model.layers
        ]

    def set_fast_weight_mode(self, mode: str):
        for module in self._fast_weight_mlps():
            module.set_mode(mode)

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
        hidden_states = self.model(input_ids=input_ids)
        return self.bidirectional_forward(
            hidden_states,
            attention_mask,
        )

    def second_pass_forward(
        self,
        input_ids: torch.LongTensor,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.FloatTensor,
        logits_to_keep: slice,
    ) -> torch.FloatTensor:
        """Run both residual streams and return the loss stream."""
        hidden_states = self.model(
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
        for module in self._fast_weight_mlps():
            module.init_state(batch_size, device)

    @torch.no_grad()
    def accumulate_gradients(self):
        for module in self._fast_weight_mlps():
            module.accumulate_gradients()

    @torch.no_grad()
    def update_state(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ):
        for module in self._fast_weight_mlps():
            module.update_state(
                embeddings,
                embedding_mask,
            )

    @torch.no_grad()
    def finalize_gradients(self):
        for module in self._fast_weight_mlps():
            module.finalize_gradients()

    @torch.no_grad()
    def empty_state(self):
        for module in self._fast_weight_mlps():
            module.empty_state()

    @torch.no_grad()
    def relative_grad_error(self) -> torch.FloatTensor:
        errors = [
            module.relative_grad_error()
            for module in self._fast_weight_mlps()
        ]

        if not errors:
            return torch.zeros(
                (),
                device=self.model.embed_tokens.weight.device,
            )
        return torch.stack(errors).mean()
