import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from omegaconf import DictConfig
from enum import Enum

from transformers.activations import ACT2FN

from models.llama import LlamaDecoderLayer, LlamaForCausalLM
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import gaussian_init, fixed_linear, unit_softplus, safe_copy_state, shift
from utils.torch_modules import SoftPass, ResidualConvMixer


def _get_update(
    activations: torch.FloatTensor,
    output_grad: torch.FloatTensor,
    gate: torch.FloatTensor,
    mask: torch.Tensor,
    down_weight: torch.FloatTensor,
    eps: float,
    lr: torch.FloatTensor,
    token_gate_logits: torch.FloatTensor,
    a_scale_logits: torch.FloatTensor,
    g_scale_logits: torch.FloatTensor,
) -> torch.FloatTensor:

    def _vnorm(x):
        return F.rms_norm(x, x.shape[-1:], eps=eps)

    # use float32 to avoid gradient accumulation drift
    mask = mask[..., None].float()
    valid_count = mask.sum(dim=-2, keepdim=True).clamp_min(1.0)
    
    activations = activations.float() * mask
    output_grad = output_grad.float() * mask
    gate = gate.float() * mask
    down_weight = down_weight.float()
    lr = lr.float()
    token_gate_logits = token_gate_logits.float() * mask
    a_scale_logits = a_scale_logits.float() * mask
    g_scale_logits = g_scale_logits.float() * mask

    # raw gradient
    g_raw = fixed_linear(output_grad, down_weight.T)

    G = torch.einsum("blo,bli->boi", g_raw * gate, activations)

    # normalize over sequence dimension
    a_norm = _sequence_rms(activations, valid_count, eps)
    g_norm = _sequence_rms(g_raw, valid_count, eps)

    # apply the gates to g
    g_gated = g_norm * _vnorm(gate)

    # apply the scales
    a_scaled = a_norm * _vnorm(unit_softplus(a_scale_logits))
    g_scaled = g_gated * _vnorm(unit_softplus(g_scale_logits))

    # learned token gate
    a_token = a_scaled * unit_softplus(token_gate_logits)

    # learned update
    update = torch.einsum("blo,bli->boi", g_scaled, a_token)
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
        gate: torch.FloatTensor,
        valid_mask: torch.Tensor,
        down_weight: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        state_buffer: torch.FloatTensor,
        grad_eps: float,
        mode: str,
        *updaters,
    ) -> torch.FloatTensor:

        ctx.save_for_backward(
            activations,
            gate,
            valid_mask,
            down_weight,
            grad_buffer,
            *updaters,
        )

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

        (
            activations,
            gate,
            valid_mask,
            down_weight,
            grad_buffer,
            *updaters
        ) = ctx.saved_tensors

        if mode == PianoMode.TRAIN_FIRST:

            G, update = _get_update(
                activations,
                output_grad,
                gate,
                valid_mask,
                down_weight,
                grad_eps,
                *updaters
            )

            return ( 
                None, # activations
                output_grad, # output
                None, # gate
                None, # valid_mask
                None, # down_weight
                G, # grad_buffer
                update, # state_buffer
                None, # grad_eps
                None, # mode
                *[None for _ in updaters]
            )

        elif mode != PianoMode.TRAIN_SECOND:
            raise RuntimeError(f"invalid fast-weight mode in backward: {mode}")

        with torch.enable_grad():
            activations_leaf = _get_leaf(activations)
            gate_leaf = _get_leaf(gate)
            down_weight_leaf = _get_leaf(down_weight)
            update_leafs = [_get_leaf(updater) for updater in updaters]

            G, update = _get_update(
                activations_leaf,
                output_grad,
                gate_leaf,
                valid_mask,
                down_weight_leaf,
                grad_eps,
                *update_leafs
            )

            local_loss = (
                update * (grad_buffer - G).detach()
            ).sum()

            (
                activation_grad,
                gate_grad,
                down_weight_grad,
                *updater_grads
            )  = torch.autograd.grad(
                local_loss,
                (
                    activations_leaf,
                    gate_leaf,
                    down_weight_leaf,
                    *update_leafs
                ),
            )

        return (
            activation_grad.to(activations.dtype),
            output_grad,
            gate_grad.to(gate.dtype),
            None, # valid_mask
            down_weight_grad.to(down_weight.dtype),
            G, # grad_buffer
            update, # state_buffer
            None, # grad_eps
            None, # mode
            *[g.to(u.dtype) for g, u in zip(updater_grads, updaters)]
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

        # sequence message passer
        self.passer = SoftPass(
            config.hidden_size,
            config.pool_size
        )

        # scale projections
        self.a_scale_proj = nn.Linear(
            config.hidden_size,
            config.fast_weight_size,
            bias=True
        )
        self.a_scale_mixer = ResidualConvMixer(
            config.fast_weight_size,
            config.mixer_size,
        )
        self.g_scale_proj = nn.Linear(
            config.hidden_size,
            config.fast_weight_size,
            bias=True
        )
        self.g_scale_mixer = ResidualConvMixer(
            config.fast_weight_size,
            config.mixer_size,
        )

        self.a_scale_pass_proj = nn.Linear(
            config.pool_size,
            config.fast_weight_size,
            bias=False
        )
        self.g_scale_pass_proj = nn.Linear(
            config.pool_size,
            config.fast_weight_size,
            bias=False
        )

        # token gate parameters
        self.token_gate_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=True,
        )
        self.token_gate_next_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=True,
        )
        self.token_gate_pass_proj = nn.Linear(
            config.pool_size,
            1,
            bias=False,
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

        messages = self.passer(hidden_states, valid_mask)

        token_gate_logits = (
            self.token_gate_proj(hidden_states) +
            self.token_gate_next_proj(shift(hidden_states, 1, 1, "left", True)) +
            self.token_gate_pass_proj(messages)
        ) / math.sqrt(3)

        a_scale_logits = (
            self.a_scale_mixer(
                self.a_scale_proj(hidden_states), valid_mask
            ) +
            self.a_scale_pass_proj(messages)
        ) / math.sqrt(2)
        g_scale_logits = (
            self.g_scale_mixer(
                self.g_scale_proj(hidden_states), valid_mask
            ) +
            self.g_scale_pass_proj(messages)
        ) / math.sqrt(2)

        return lr, token_gate_logits, a_scale_logits, g_scale_logits


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
            bias=False,
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

        # fast mlp
        activations = self.fast_act_fn(self.up_fast(x), self.gate_fast(x))

        if mode == PianoMode.INFERENCE:
            v = torch.einsum(
                "boi,bli->blo", self.state.detach(), activations
            )
            gate = 2 * torch.sigmoid(self.sig_fast(x))
            output = self.down_fast(v * gate)

            return y_base + output

        if valid_mask is None:
            valid_mask = torch.ones_like(x[..., 0], dtype=torch.bool)

        if mode == PianoMode.TRAIN_FIRST:

            v = torch.einsum(
                "boi,bli->blo", self.state.detach(), activations
            )
            updaters = self.fast_dynamic_lr(x, valid_mask)

            gate = 2 * torch.sigmoid(self.sig_fast(x))
            output = self.down_fast(v * gate)

            output = PianoFastWeightFunction.apply(
                activations,
                output,
                gate,
                valid_mask,
                self.down_fast.weight,
                self.grad_buffer,
                self.state,
                self.grad_eps,
                mode,
                *updaters,
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
            updaters = self.fast_dynamic_lr(x_prop, valid_mask)

            gate = 2 * torch.sigmoid(self.sig_fast(x))
            gate_replay = maybe_shard_with_gradients(gate[::2])
            gate_prop = maybe_shard_with_gradients(gate[1::2])

            output_replay = self.down_fast(v_replay * gate_replay)
            output_prop = self.down_fast(v_prop * gate_prop)

            output_replay = PianoFastWeightFunction.apply(
                act_prop,
                output_replay,
                gate_prop,
                valid_mask,
                self.down_fast.weight,
                self.grad_buffer,
                self.state,
                self.grad_eps,
                mode,
                *updaters,
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
            gaussian_init(mlp)

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
