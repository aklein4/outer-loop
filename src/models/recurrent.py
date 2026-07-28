import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from omegaconf import DictConfig
from enum import Enum

from transformers.activations import ACT2FN

from models.layers import BidirectionalHead
from models.llama import (
    LlamaForCausalLM,
    LlamaRMSNorm,
)
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import fixed_linear, gaussian_init


def _get_G(
    activations: torch.FloatTensor,
    output_grad: torch.FloatTensor,
    down_weight: torch.FloatTensor,
) -> torch.FloatTensor:
    # all float to avoid gradient accumulation drift
    value_gradient = fixed_linear(
        output_grad.float(), down_weight.T.float()
    )
    return value_gradient.mT @ activations.float()


def _get_leaf(x: torch.FloatTensor) -> torch.FloatTensor:
    return x.detach().requires_grad_(True)


class RecurrentMode(Enum):
    INFERENCE = "inference"
    TRAIN_FIRST = "train_first"
    TRAIN_SECOND = "train_second"


_MODE_NUM_ELEMENTS = {
    RecurrentMode.INFERENCE: 1,
    RecurrentMode.TRAIN_FIRST: 2,
    RecurrentMode.TRAIN_SECOND: 3,
}


def _mode_to_tensor(
    mode: RecurrentMode,
    reference: torch.Tensor,
) -> torch.Tensor:
    try:
        num_elements = _MODE_NUM_ELEMENTS[mode]
    except KeyError:
        raise ValueError(f"unknown recurrent mode: {mode}") from None
    return reference.new_zeros(num_elements)


def _tensor_to_mode(mode_tensor: torch.Tensor) -> RecurrentMode:
    num_elements = mode_tensor.numel()
    for mode, expected_num_elements in _MODE_NUM_ELEMENTS.items():
        if num_elements == expected_num_elements:
            return mode
    raise ValueError(
        f"mode tensor num elements must be in {list(_MODE_NUM_ELEMENTS.values())}, "
        f"got {num_elements}"
    )


class RecurrentFastWeightFunction(torch.autograd.Function):
    """Collect raw fast-weight gradients and inject the local FO gradient."""

    @staticmethod
    def forward(
        ctx,
        activations: torch.FloatTensor,
        output: torch.FloatTensor,
        down_weight: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        lr: torch.FloatTensor | None,
        grad_eps: float,
        mode: str,
    ) -> torch.FloatTensor:

        to_save = (
            activations,
            down_weight,
        )
        if mode == RecurrentMode.TRAIN_SECOND:
            to_save += (grad_buffer, lr)

        ctx.save_for_backward(*to_save)
        ctx.grad_eps = grad_eps
        ctx.mode = mode

        return output


    @staticmethod
    def backward(
        ctx,
        output_grad: torch.FloatTensor,
    ):
        grad_eps = ctx.grad_eps
        mode = ctx.mode

        if mode == RecurrentMode.TRAIN_FIRST:
            activations, down_weight = ctx.saved_tensors

            G = _get_G(
                activations,
                output_grad,
                down_weight,
            )

            return (
                None,
                output_grad,
                None,
                G,
                None,
                None,
                None
            )

        elif mode != RecurrentMode.TRAIN_SECOND:
            raise RuntimeError(f"invalid fast-weight mode in backward: {mode}")

        (
            activations,
            down_weight,
            remaining_grad, # grad_buffer
            lr,
        ) = ctx.saved_tensors

        with torch.enable_grad():
            activations_leaf = _get_leaf(activations)
            down_weight_leaf = _get_leaf(down_weight)
            lr_leaf = _get_leaf(lr)

            G = _get_G(
                activations_leaf,
                output_grad,
                down_weight_leaf,
            )

            future_grad = (
                remaining_grad - G
            ).detach()

            with torch.autocast(
                str(future_grad.device.type),
                dtype=torch.bfloat16,
            ):

                G_normed = F.rms_norm(
                    G.float(), G.shape[-2:], eps=grad_eps,
                )
                state_update = -lr_leaf * G_normed

                local_loss = (future_grad * state_update).sum()

            (
                activation_grad,
                down_weight_grad,
                lr_grad,
            ) = torch.autograd.grad(
                local_loss,
                (activations_leaf, down_weight_leaf, lr_leaf),
            )

        return (
            activation_grad.to(activations.dtype),
            output_grad,
            down_weight_grad.to(down_weight.dtype),
            G,
            lr_grad.to(lr.dtype),
            None,
            None,
        )


def unit_glu(x: torch.FloatTensor, gate: torch.FloatTensor) -> torch.FloatTensor:
    return x * F.silu(gate) / 0.6


class DynamicLR(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.fast_weight_size = config.fast_weight_size

        self.base_lr = config.base_lr

        self.scalar_scaler = math.sqrt(self.fast_weight_size)
        self.rms_norm_eps = config.rms_norm_eps

        # learning-rate parameters
        self.fast_log_lr = nn.Parameter(
            torch.randn(self.fast_weight_size, self.fast_weight_size)
            * 0.35 / self.scalar_scaler
        )
        self.fast_m = nn.Parameter(
            torch.ones(self.fast_weight_size, self.fast_weight_size)
            * 0.35 / self.scalar_scaler
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

        self.fast_attn_r = nn.Linear(
            config.hidden_size, 1, bias=False
        )
        self.fast_attn_l = nn.Linear(
            config.hidden_size, 1, bias=False
        )


    def forward(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:

        embedding_mask = embedding_mask.to(embeddings.dtype)
        masked_embeddings = embeddings * embedding_mask[..., None]

        a_r = self.fast_attn_r(masked_embeddings)
        a_r = torch.masked_fill(a_r, embedding_mask[..., None] < 0.5, -100.0)
        attn_r = F.softmax(a_r, dim=-2)

        a_l = self.fast_attn_l(masked_embeddings)
        a_l = torch.masked_fill(a_l, embedding_mask[..., None] < 0.5, -100.0)
        attn_l = F.softmax(a_l, dim=-2)

        l = (self.fast_p_l(masked_embeddings) * attn_l).sum(dim=-2)
        l = F.rms_norm(l, l.shape[-1:], eps=self.rms_norm_eps)

        r = (self.fast_p_r(masked_embeddings) * attn_r).sum(dim=-2)
        r = F.rms_norm(r, r.shape[-1:], eps=self.rms_norm_eps)

        offset = (
            l[:, :, None] + r[:, None, :]
        ) * (self.fast_m[None] * self.scalar_scaler)

        log_lr = (
            self.fast_log_lr * self.scalar_scaler
            + offset
        )

        return (
            torch.exp(log_lr)
            * self.base_lr
            / math.sqrt(self.fast_weight_size)
        )


class RecurrentFastWeightMLP(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        # save config
        self.intermediate_size = config.intermediate_size
        self.fast_weight_size = config.fast_weight_size

        self.grad_eps = config.grad_rms_eps

        self.act_fn = ACT2FN[config.hidden_act]

        # base projections
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        # fast projections
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

        self.dynamic_lr = DynamicLR(config)

        # ephemeral state
        self.state: nn.Buffer
        self.grad_buffer: nn.Buffer
        self.final_grad_norm: nn.Buffer


    @torch.no_grad()
    def pretrained_init(self) -> None:
        if self.fast_weight_size > self.intermediate_size:
            raise ValueError(
                "fast_weight_size must not exceed intermediate_size"
            )
        s = self.fast_weight_size

        self.up_fast.weight.copy_(self.up_proj.weight[:s])
        self.gate_fast.weight.copy_(self.gate_proj.weight[:s])

        self.down_fast.weight.copy_(self.down_proj.weight[:, :s])

    
    def forward(
        self,
        x: torch.FloatTensor,
        fast_weight_mode: torch.Tensor | None = None,
        lr_embeddings: torch.FloatTensor | None = None,
        lr_embedding_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:

        mode = RecurrentMode.INFERENCE
        if fast_weight_mode is not None:
            mode = _tensor_to_mode(fast_weight_mode)

        # base mlp
        h_base = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        y_base = self.down_proj(h_base)

        # fast mlp
        h_fast = unit_glu(
            self.up_fast(x),
            self.gate_fast(x),
        )
        value = torch.einsum("boi,bli->blo", self.state, h_fast)
        output = self.down_fast(value)

        activations = h_fast.detach()
        lr = None
        if mode == RecurrentMode.TRAIN_SECOND:

            x_d = x.detach()
            activations = unit_glu(
                self.up_fast(x_d),
                self.gate_fast(x_d),
            )

            lr = self.dynamic_lr(lr_embeddings, lr_embedding_mask)

        if mode != RecurrentMode.INFERENCE:
            output = RecurrentFastWeightFunction.apply(
                activations,
                output,
                self.down_fast.weight,
                self.grad_buffer,
                lr,
                self.grad_eps,
                mode,
            )

        return y_base + output


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device) -> None:

        state = torch.zeros(
            bs, self.fast_weight_size, self.fast_weight_size,
            device=device, dtype=torch.float32
        )
        grad_buffer = torch.zeros_like(state)
        final_grad_norm = torch.zeros(
            bs, device=device, dtype=torch.float32
        )

        state = maybe_shard_with_gradients(state)
        grad_buffer = maybe_shard_with_gradients(grad_buffer)
        final_grad_norm = maybe_shard_with_gradients(final_grad_norm)

        self.register_buffer("state", state, persistent=False)
        self.register_buffer("grad_buffer", grad_buffer, persistent=False)
        self.register_buffer("final_grad_norm", final_grad_norm, persistent=False)

        self.state.requires_grad_(False)

        # backward stores G in grad_buffer.grad
        self.grad_buffer.requires_grad_(True)
        self.grad_buffer.grad = maybe_shard_with_gradients(
            torch.zeros_like(self.grad_buffer)
        )

        self.final_grad_norm.requires_grad_(False)


    @torch.no_grad()
    def update_state(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
        mode: RecurrentMode,
    ) -> None:

        G = self.grad_buffer.grad
        
        G_norm = F.rms_norm(
            G.float(), G.shape[-2:], eps=self.grad_eps,
        )

        lr = self.dynamic_lr(embeddings, embedding_mask)

        state_update = -lr * G_norm

        self.state.add_(state_update)

        if mode == RecurrentMode.TRAIN_FIRST:
            self.grad_buffer.add_(G)
        elif mode == RecurrentMode.TRAIN_SECOND:
            self.grad_buffer.sub_(G)
        else:
            raise ValueError(f"invalid state update mode: {mode}")

        self.grad_buffer.grad.zero_()


    @torch.no_grad()
    def finalize_state(self):

        self.state.zero_()

        self.final_grad_norm.copy_(
            self.grad_buffer.norm(dim=(-2, -1))
        )

        self.grad_buffer.grad.zero_()


    @torch.no_grad()
    def empty_state(self) -> None:

        self.state.zero_()

        self.grad_buffer.zero_()
        self.grad_buffer.grad.zero_()

        self.final_grad_norm.zero_()


    @torch.no_grad()
    def relative_grad_error(self) -> torch.FloatTensor:
        return (
            self.grad_buffer.norm(dim=(-2, -1))
            / (self.final_grad_norm + self.grad_eps)
        )



class RecurrentModel(LlamaForCausalLM):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        # bidirectional head for lr embeddings
        self.embedding_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.bidirectional_head = BidirectionalHead(config)

        self.disable_fast_weights = config.get(
            "disable_fast_weights",
            False,
        )
        if not self.disable_fast_weights:
            for layer in self.model.layers:
                layer.mlp = RecurrentFastWeightMLP(config)

        self.apply(gaussian_init)


    def _old_load_state_dict(
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

        for mlp in self.fast_modules():
            mlp.pretrained_init()

        return incompatible_keys
    

    def forward_backbone(
        self,
        input_ids: torch.LongTensor,
        embeddings: torch.FloatTensor | None = None,
        embedding_mask: torch.BoolTensor | None = None,
        mode: RecurrentMode | None = None,
    ) -> torch.FloatTensor:
        if self.disable_fast_weights:
            return self.model(input_ids=input_ids)

        model_kwargs = {}
        if mode is not None:
            model_kwargs["fast_weight_mode"] = _mode_to_tensor(
                mode,
                self.model.embed_tokens.weight,
            )

        if embeddings is not None:
            model_kwargs["lr_embeddings"] = embeddings
        if embedding_mask is not None:
            model_kwargs["lr_embedding_mask"] = embedding_mask.float()

        return self.model(
            input_ids=input_ids,
            **model_kwargs,
        )


    def forward_lm(
        self,
        hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.lm_head(
            self.model.norm(hidden_states)
        )


    def forward_embeddings(
        self,
        hidden_states: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        hidden_states = self.embedding_norm(hidden_states)
        return self.bidirectional_head(hidden_states, embedding_mask)


    def _layer_module(self, layer, name: str) -> nn.Module:
        try:
            return layer.get_submodule(name)
        except AttributeError:
            return layer._orig_mod.get_submodule(name)


    def fast_modules(self) -> list[RecurrentFastWeightMLP]:
        if self.disable_fast_weights:
            return []
        return [
            self._layer_module(layer, "mlp") for layer in self.model.layers
        ]


    def grad_buffers(self) -> list[torch.FloatTensor]:
        return [
            mlp.grad_buffer for mlp in self.fast_modules()
        ]


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device) -> None:
        for mlp in self.fast_modules():
            mlp.init_state(bs, device)


    @torch.no_grad()
    def update_state(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
        mode: RecurrentMode,
    ) -> None:
        for mlp in self.fast_modules():
            mlp.update_state(embeddings, embedding_mask, mode)


    @torch.no_grad()
    def finalize_state(self) -> None:
        for mlp in self.fast_modules():
            mlp.finalize_state()


    @torch.no_grad()
    def empty_state(self) -> None:
        for mlp in self.fast_modules():
            mlp.empty_state()
    

    @torch.no_grad()
    def relative_grad_error(self) -> torch.FloatTensor:
        if self.disable_fast_weights:
            return 0.0
        
        errors = [
            mlp.relative_grad_error() for mlp in self.fast_modules()
        ]
        return torch.stack(errors).mean()
