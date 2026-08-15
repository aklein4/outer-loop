import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from omegaconf import DictConfig
from enum import Enum

from transformers.activations import ACT2FN

from models.llama import LlamaDecoderLayer, LlamaForCausalLM
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import fixed_linear, unit_softplus, safe_copy_state, shift
from utils.torch_modules import SoftPool


def _get_update(
    activations: torch.FloatTensor,
    output_grad: torch.FloatTensor,
    down_weight: torch.FloatTensor,
    output_gate: torch.FloatTensor,
    update_output_gate_logits: torch.FloatTensor,
    token_gate_logits: torch.FloatTensor,
    lr: torch.FloatTensor,
    valid_mask: torch.BoolTensor,
    eps: float,
) -> torch.FloatTensor:

    # use float32 to avoid gradient accumulation drift
    mask = valid_mask[..., None].float()
    valid_count = mask.sum(dim=-2, keepdim=True).clamp_min(1.0)
    
    activations = activations.float() * mask
    output_grad = output_grad.float() * mask
    down_weight = down_weight.float()
    output_gate = output_gate.float()
    update_output_gate_logits = update_output_gate_logits.float()
    token_gate_logits = token_gate_logits.float()

    projected_grad = fixed_linear(output_grad, down_weight.T)
    forward_g = projected_grad * output_gate
    update_g = projected_grad * _sigmoid_rms(update_output_gate_logits, eps)

    G = torch.einsum("blo,bli->boi", forward_g, activations)

    # normalize over sequence dimension
    a_norm = _sequence_rms(activations, valid_count, eps)
    g_norm = _sequence_rms(update_g, valid_count, eps)

    # learned token gate
    a_gated = a_norm * unit_softplus(token_gate_logits)

    update = torch.einsum("blo,bli->boi", g_norm, a_gated)
    update = -update * lr

    return G, update


def _sequence_rms(
    x: torch.FloatTensor,
    valid_count: torch.FloatTensor,
    eps: float,
):
    return x * torch.rsqrt(
        x.square().sum(dim=-2, keepdim=True) / valid_count
        + eps**2
    )


def _sigmoid_rms(
    logits: torch.FloatTensor,
    eps: float,
):
    x = 2 * torch.sigmoid(logits)
    return F.rms_norm(x, x.shape[-1:], eps=eps)


def _get_leaf(x: torch.FloatTensor) -> torch.FloatTensor:
    return x.detach().requires_grad_(True)


class PianoMode(Enum):
    INFERENCE = "inference"
    TRAIN_FIRST = "train_first"
    TRAIN_SECOND = "train_second"


# can't pass non-tensor objects to torch-xla scanned layers
# but can't branch based on tensor value
# so we encode the mode as a tensor with a different number of elements for each mode 
_MODE_NUM_ELEMENTS = {
    PianoMode.INFERENCE: 1,
    PianoMode.TRAIN_FIRST: 2,
    PianoMode.TRAIN_SECOND: 3,
}


def _mode_to_tensor(
    mode: PianoMode,
    reference: torch.Tensor,
) -> torch.Tensor:
    try:
        num_elements = _MODE_NUM_ELEMENTS[mode]
    except KeyError:
        raise ValueError(f"unknown forte mode: {mode}") from None
    return reference.new_zeros(num_elements)

def _tensor_to_mode(mode_tensor: torch.Tensor) -> PianoMode:
    num_elements = mode_tensor.numel()
    for mode, expected_num_elements in _MODE_NUM_ELEMENTS.items():
        if num_elements == expected_num_elements:
            return mode
    raise ValueError(
        f"mode tensor num elements must be in {list(_MODE_NUM_ELEMENTS.values())}, "
        f"got {num_elements}"
    )


class PianoFastWeightFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        activations: torch.FloatTensor,
        output: torch.FloatTensor,
        down_weight: torch.FloatTensor,
        output_gate: torch.FloatTensor,
        update_output_gate_logits: torch.FloatTensor,
        token_gate_logits: torch.FloatTensor,
        valid_mask: torch.BoolTensor,
        grad_buffer: torch.FloatTensor,
        state_buffer: torch.FloatTensor,
        lr: torch.FloatTensor | None,
        grad_eps: float,
        mode: str,
    ) -> torch.FloatTensor:

        to_save = (
            activations,
            down_weight,
            output_gate,
            update_output_gate_logits,
            token_gate_logits,
            valid_mask,
            lr,
        )
        if mode == PianoMode.TRAIN_SECOND:
            to_save += (grad_buffer,)

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

        if mode == PianoMode.TRAIN_FIRST:
            (
                activations,
                down_weight,
                output_gate,
                update_output_gate_logits,
                token_gate_logits,
                valid_mask,
                lr
            ) = ctx.saved_tensors

            G, update = _get_update(
                activations,
                output_grad,
                down_weight,
                output_gate,
                update_output_gate_logits,
                token_gate_logits,
                lr,
                valid_mask,
                grad_eps,
            )

            return ( 
                None, # activations
                output_grad, # output
                None, # down_weight
                None, # output_gate
                None, # update_output_gate_logits
                None, # token_gate_logits
                None, # valid_mask
                G, # grad_buffer
                update, # state_buffer
                None, # lr
                None, # grad_eps
                None, # mode
            )

        elif mode != PianoMode.TRAIN_SECOND:
            raise RuntimeError(f"invalid fast-weight mode in backward: {mode}")

        (
            activations,
            down_weight,
            output_gate,
            update_output_gate_logits,
            token_gate_logits,
            valid_mask,
            lr,
            grad_buffer,
        ) = ctx.saved_tensors

        with torch.enable_grad():
            activations_leaf = _get_leaf(activations)
            down_weight_leaf = _get_leaf(down_weight)
            update_output_gate_logits_leaf = _get_leaf(update_output_gate_logits)
            token_gate_logits_leaf = _get_leaf(token_gate_logits)
            lr_leaf = _get_leaf(lr)

            G, update = _get_update(
                activations_leaf,
                output_grad,
                down_weight_leaf,
                output_gate,
                update_output_gate_logits_leaf,
                token_gate_logits_leaf,
                lr_leaf,
                valid_mask,
                grad_eps,
            )

            future_grad = (
                grad_buffer - G
            ).detach()

            with torch.autocast(
                str(future_grad.device.type),
                dtype=torch.bfloat16,
            ):

                local_loss = (
                    future_grad * update
                ).sum()

            (
                activation_grad,
                down_weight_grad,
                update_output_gate_logits_grad,
                token_gate_logits_grad,
                lr_grad,
            ) = torch.autograd.grad(
                local_loss,
                (
                    activations_leaf,
                    down_weight_leaf,
                    update_output_gate_logits_leaf,
                    token_gate_logits_leaf,
                    lr_leaf,
                ),
            )

        return (
            activation_grad.to(activations.dtype),
            output_grad,
            down_weight_grad.to(down_weight.dtype),
            None, # output_gate; G is not part of the surrogate gradient
            update_output_gate_logits_grad.to(update_output_gate_logits.dtype),
            token_gate_logits_grad.to(token_gate_logits.dtype),
            None, # valid_mask
            G, # grad_buffer
            update, # state (already scaled by -lr)
            lr_grad.to(lr.dtype),
            None, # grad_eps
            None, # mode
        )
    

class DynamicLR(nn.Module):

    no_muon_patterns = (
        "log_lr",
    )


    def __init__(self, config: DictConfig):
        super().__init__()

        self.fast_weight_size = config.fast_weight_size

        self.base_lr = config.base_lr

        self.scalar_scaler = math.sqrt(self.fast_weight_size)
        self.rms_norm_eps = config.rms_norm_eps

        # learning-rate parameters
        self.log_lr = nn.Parameter(
            torch.zeros(self.fast_weight_size, self.fast_weight_size)
        )

        self.token_gate_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
        )
        self.next_token_gate_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
        )
        self.sequence_token_gate = SoftPool(
            config.hidden_size,
            config.pool_size,
            output_size=1,
            do_norm=True,
        )


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        valid_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        hidden_states = hidden_states * valid_mask[..., None].to(hidden_states.dtype)

        lr = torch.exp(
            self.log_lr[None] * self.scalar_scaler
            + math.log(self.base_lr)
            - math.log(self.fast_weight_size)
        )

        token_gate_logits = (
            self.token_gate_proj(hidden_states) +
            self.next_token_gate_proj(
                shift(hidden_states, 1, 1, "left", True)
            ) +
            self.sequence_token_gate(hidden_states, valid_mask).unsqueeze(-2)
        ) / math.sqrt(3)

        return lr, token_gate_logits


class UnitGLU(nn.Module):
    def forward(self, x, gate):
        return x * F.silu(gate) / 0.6


class PianoFastWeightMLP(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        # save config
        self.intermediate_size = config.intermediate_size
        self.fast_weight_size = config.fast_weight_size

        self.grad_eps = config.grad_rms_eps

        self.act_fn = ACT2FN[config.hidden_act]
        self.fast_act_fn = UnitGLU()

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
        self.sig_fast = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=True
        )
        self.down_fast = nn.Linear(
            self.fast_weight_size,
            config.hidden_size,
            bias=False,
        )

        self.sig_offset_mixer = nn.Parameter(
            torch.zeros(self.fast_weight_size)
        )
        self.sig_fast_offset = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=False
        )
        self.sig_fast_offset.weight.data.zero_()
        self.sig_fast_offset.weight.inited = True

        self.fast_dynamic_lr = DynamicLR(config)

        # ephemeral state
        self.state: nn.Buffer
        self.grad_buffer: nn.Buffer
        self.final_grad_norm: nn.Buffer


    def standard_forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )


    def forward(
        self,
        x: torch.FloatTensor,
        fast_weight_mode: torch.Tensor | None = None,
        valid_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        y_base = self.standard_forward(x)

        mode = PianoMode.INFERENCE
        if fast_weight_mode is not None:
            mode = _tensor_to_mode(fast_weight_mode)

        if valid_mask is None:
            valid_mask = torch.ones_like(x[..., 0], dtype=torch.bool)

        # fast mlp
        activations = self.fast_act_fn(self.up_fast(x), self.gate_fast(x))

        if mode == PianoMode.INFERENCE:
            v = torch.einsum(
                "boi,bli->blo", self.state.detach(), activations
            )
            output_gate = 2 * torch.sigmoid(self.sig_fast(x))
            output = self.down_fast(v * output_gate)

            return y_base + output

        if mode == PianoMode.TRAIN_FIRST:

            v = torch.einsum(
                "boi,bli->blo", self.state.detach(), activations
            )
            lr, token_gate_logits = self.fast_dynamic_lr(x, valid_mask)

            output_gate_logits = self.sig_fast(x)
            output_gate = 2 * torch.sigmoid(output_gate_logits)
            output = self.down_fast(v * output_gate)

            offset_gate_logits = (
                output_gate_logits * (1.0 + self.sig_offset_mixer[None, None])
                + self.sig_fast_offset(x)
            )

            output = PianoFastWeightFunction.apply(
                activations,
                output,
                self.down_fast.weight,
                output_gate,
                offset_gate_logits,
                token_gate_logits,
                valid_mask,
                self.grad_buffer,
                self.state,
                lr,
                self.grad_eps,
                mode,
            )

            return y_base + output

        if mode == PianoMode.TRAIN_SECOND:
            act_replay = maybe_shard_with_gradients(activations[::2])
            act_prop = maybe_shard_with_gradients(activations[1::2])
            x_prop = maybe_shard_with_gradients(x[1::2])

            v_replay = torch.einsum(
                "boi,bli->blo", self.state.detach(), act_replay
            )
            v_prop = torch.einsum(
                "boi,bli->blo", self.state.detach(), act_prop
            )
            lr, token_gate_logits = self.fast_dynamic_lr(
                x_prop, valid_mask
            )

            output_gate_logits = self.sig_fast(x)
            gate_logits_replay = maybe_shard_with_gradients(output_gate_logits[::2])
            gate_logits_prop = maybe_shard_with_gradients(output_gate_logits[1::2])

            gate_replay = 2 * torch.sigmoid(gate_logits_replay)
            gate_prop = 2 * torch.sigmoid(gate_logits_prop)

            output_replay = self.down_fast(v_replay * gate_replay)
            output_prop = self.down_fast(v_prop * gate_prop)

            offset_gate_logits = (
                gate_logits_prop * (1.0 + self.sig_offset_mixer[None, None])
                + self.sig_fast_offset(x_prop)
            )

            output_replay = PianoFastWeightFunction.apply(
                act_prop,
                output_replay,
                self.down_fast.weight,
                gate_prop,
                offset_gate_logits,
                token_gate_logits,
                valid_mask,
                self.grad_buffer,
                self.state,
                lr,
                self.grad_eps,
                mode,
            )

            output = maybe_shard_with_gradients(
                torch.stack(
                    [output_replay, output_prop], dim=1
                ).reshape(-1, *output_replay.shape[1:])
            )

            return y_base + output

        raise ValueError(f"invalid fast weight mode: {mode}")


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

        # stores the update
        self.state.requires_grad_(True)
        self.state.grad = maybe_shard_with_gradients(
            torch.zeros_like(self.state)
        )

        # backward stores G in grad_buffer.grad
        self.grad_buffer.requires_grad_(True)
        self.grad_buffer.grad = maybe_shard_with_gradients(
            torch.zeros_like(self.grad_buffer)
        )

        self.final_grad_norm.requires_grad_(False)


    @torch.no_grad()
    def update_state(
        self,
        mode: PianoMode,
        lr_scale: float = 1.0
    ) -> None:

        G = self.grad_buffer.grad
        update = self.state.grad
        
        self.state.add_(update * lr_scale)

        if mode == PianoMode.TRAIN_FIRST:
            self.grad_buffer.add_(G)
        elif mode == PianoMode.TRAIN_SECOND:
            self.grad_buffer.sub_(G)
        else:
            raise ValueError(f"invalid state update mode: {mode}")

        self.grad_buffer.grad.zero_()
        self.state.grad.zero_()


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


class PianoModel(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)

        for layer in self.model.layers._iter_layers():
            layer: LlamaDecoderLayer

            mlp = PianoFastWeightMLP(config)
            safe_copy_state(layer.mlp, mlp, strict=False)
            layer.mlp = mlp


    def forward(self, *args, mode=PianoMode.INFERENCE, **kwargs):
        if "input_ids" in kwargs:
            ref = kwargs["input_ids"]
        elif "inputs_embeds" in kwargs:
            ref = kwargs["inputs_embeds"]
        elif len(args) > 0:
            ref = args[0]
        else:
            raise ValueError("must provide input_ids or inputs_embeds")
        fast_weight_mode = _mode_to_tensor(mode, ref)

        return super().forward(*args, fast_weight_mode=fast_weight_mode, **kwargs)


    def fast_modules(self):
        for module in self.modules():
            if isinstance(module, PianoFastWeightMLP):
                yield module

    def grad_containers(self) -> list[torch.FloatTensor]:
        return [
            mlp.grad_buffer for mlp in self.fast_modules()
        ] + [
            mlp.state for mlp in self.fast_modules()
        ]
    

    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for module in self.fast_modules():
            module.init_state(bs, device)

    @torch.no_grad()
    def update_state(
        self,
        mode: PianoMode,
        **kwargs
    ) -> None:
        for module in self.fast_modules():
            module.update_state(mode, **kwargs)

    @torch.no_grad()
    def finalize_state(self):
        for module in self.fast_modules():
            module.finalize_state()

    @torch.no_grad()
    def empty_state(self):
        for module in self.fast_modules():
            module.empty_state()


    @torch.no_grad()
    def relative_grad_error(self) -> torch.FloatTensor:
        errors = [
            mlp.relative_grad_error() for mlp in self.fast_modules()
        ]
        return torch.stack(errors).mean()
