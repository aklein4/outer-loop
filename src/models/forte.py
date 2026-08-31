import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from omegaconf import DictConfig
from enum import Enum

from transformers.activations import ACT2FN

from models.layers import BidirectionalHead
from models.llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_modules import LayerStack
from utils.torch_utils import fixed_linear, gaussian_init, unit_softplus

from torchprime.rope.rope import RopeScaling



def _get_G(
    activations: torch.FloatTensor,
    output_grad: torch.FloatTensor,
    down_weight: torch.FloatTensor,
    valid_mask: torch.BoolTensor,
    *dynamic_args: torch.FloatTensor,
    eps: float,
    offset_alpha: float,
) -> torch.FloatTensor:

    # use float32 to avoid gradient accumulation drift
    mask = valid_mask[..., None].float()
    valid_count = mask.sum(dim=-2, keepdim=True).clamp_min(1.0)

    activations = activations.float() * mask
    output_grad = output_grad.float() * mask
    down_weight = down_weight.float()
    (
        Mx,
        offset,
        lr,
        offset_lr,
        gradient_gate_logits,
        offset_gate_logits
    ) = [d.float() for d in dynamic_args]
    
    # output gradient of fast weights
    g = fixed_linear(
        output_grad, down_weight.T
    )

    # raw gradient update
    G = torch.einsum("blo,bli->boi", g, activations)

    # gradient-based update
    # normalize over sequence dimension
    a_norm = _sequence_rms(activations, valid_count, eps)
    g_norm = _sequence_rms(g, valid_count, eps)
    # learned per-token gates
    g_gated = gate_heads(g_norm, unit_softplus(gradient_gate_logits))
    # matrix
    update = torch.einsum("blo,bli->boi", g_gated, a_norm)
    update = -update * lr
    
    # offset-based update
    offset_delta = (offset - Mx) * unit_softplus(offset_gate_logits)
    offset_update = torch.einsum("blo,bli->boi", offset_delta, a_norm)
    offset_update = offset_update * offset_lr

    # cap the RMS of the offset to ensure descent
    update_rms = torch.norm(update, dim=(-2, -1), keepdim=True)
    offset_update_rms = torch.norm(
        offset_update, dim=(-2, -1), keepdim=True
    )
    offset_update_scaled = (
        offset_update
        * offset_alpha
        * torch.tanh(offset_update_rms / (update_rms + eps))
        * update_rms / (offset_update_rms + eps)
    )

    update = update + offset_update_scaled

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


def gate_heads(
    hidden_states: torch.FloatTensor,
    gates: torch.FloatTensor,
):
    if hidden_states.shape[-1] % gates.shape[-1] != 0:
        raise ValueError(
            f"hidden_states last dimension ({hidden_states.shape[-1]}) "
            f"must be divisible by gates last dimension ({gates.shape[-1]})"
        )

    h = hidden_states.view(*hidden_states.shape[:-1], gates.shape[-1], -1)
    g_h = h * gates[..., None]

    return g_h.view(*hidden_states.shape)


def _get_leaf(x: torch.FloatTensor) -> torch.FloatTensor:
    return x.detach().requires_grad_(True)


class ForteMode(Enum):
    INFERENCE = "inference"
    TRAIN_FIRST = "train_first"
    TRAIN_SECOND = "train_second"


# can't pass non-tensor objects to torch-xla scanned layers
# but can't branch based on tensor value
# so we encode the mode as a tensor with a different number of elements for each mode 
_MODE_NUM_ELEMENTS = {
    ForteMode.INFERENCE: 1,
    ForteMode.TRAIN_FIRST: 2,
    ForteMode.TRAIN_SECOND: 3,
}


def _mode_to_tensor(
    mode: ForteMode,
    reference: torch.Tensor,
) -> torch.Tensor:
    try:
        num_elements = _MODE_NUM_ELEMENTS[mode]
    except KeyError:
        raise ValueError(f"unknown forte mode: {mode}") from None
    return reference.new_zeros(num_elements)

def _tensor_to_mode(mode_tensor: torch.Tensor) -> ForteMode:
    num_elements = mode_tensor.numel()
    for mode, expected_num_elements in _MODE_NUM_ELEMENTS.items():
        if num_elements == expected_num_elements:
            return mode
    raise ValueError(
        f"mode tensor num elements must be in {list(_MODE_NUM_ELEMENTS.values())}, "
        f"got {num_elements}"
    )


class ForteFastWeightFunction(torch.autograd.Function):
    """Collect raw fast-weight gradients and inject the local FO gradient."""

    @staticmethod
    def forward(
        ctx,
        activations: torch.FloatTensor,
        output: torch.FloatTensor,
        down_weight: torch.FloatTensor,
        valid_mask: torch.BoolTensor,
        grad_buffer: torch.FloatTensor,
        state: torch.FloatTensor,
        mode: str,
        update_kwargs,
        *dynamic_args: torch.FloatTensor,
    ) -> torch.FloatTensor:

        ctx.save_for_backward(
            activations,
            down_weight,
            valid_mask,
            grad_buffer,
            *dynamic_args,
        )
        ctx.update_kwargs = update_kwargs
        ctx.mode = mode

        return output


    @staticmethod
    def backward(
        ctx,
        output_grad: torch.FloatTensor,
    ):
        mode = ctx.mode
        update_kwargs = ctx.update_kwargs

        if mode == ForteMode.TRAIN_FIRST:
            (
                activations,
                down_weight,
                valid_mask,
                grad_buffer,
                *dynamic_args
            ) = ctx.saved_tensors

            G, update = _get_G(
                activations,
                output_grad,
                down_weight,
                valid_mask,
                *dynamic_args,
                **update_kwargs,
            )

            return ( 
                None, # activations
                output_grad, # output
                None, # down_weight
                None, # valid_mask
                G, # grad_buffer
                update, # state
                None, # mode
                None, # update_kwargs
                *[None for _ in dynamic_args],
            )

        elif mode != ForteMode.TRAIN_SECOND:
            raise RuntimeError(f"invalid fast-weight mode in backward: {mode}")

        (
            activations,
            down_weight,
            valid_mask,
            grad_buffer,
            *dynamic_args
        ) = ctx.saved_tensors

        with torch.enable_grad():
            activations_leaf = _get_leaf(activations)
            down_weight_leaf = _get_leaf(down_weight)
            dynamic_leaves = [_get_leaf(arg) for arg in dynamic_args]

            G, update = _get_G(
                activations_leaf,
                output_grad,
                down_weight_leaf,
                valid_mask,
                *dynamic_leaves,
                **update_kwargs,
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
                *dynamic_grads,
            ) = torch.autograd.grad(
                local_loss,
                (
                    activations_leaf,
                    down_weight_leaf,
                    *dynamic_leaves,
                ),
            )

        return (
            activation_grad.to(activations.dtype),
            output_grad,
            down_weight_grad.to(down_weight.dtype),
            None, # valid_mask
            G, # grad_buffer
            update.detach(), # state (already scaled by -lr)
            None, # mode
            None, # update_kwargs
            *dynamic_grads,
        )


class DynamicLR(nn.Module):

    no_muon_patterns = (
        "log_lr",
        "offset_log_lr",
        "gradient_gate_proj",
    )

    def __init__(self, config: DictConfig):
        super().__init__()

        self.fast_weight_size = config.fast_weight_size

        self.base_lr = config.base_lr
        self.num_fast_weight_heads = config.num_fast_weight_heads

        self.scalar_scaler = math.sqrt(self.fast_weight_size)
        self.rms_norm_eps = config.rms_norm_eps

        # learning-rate parameters
        self.offset_proj = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=False,
        )

        self.log_lr = nn.Parameter(
            torch.zeros(self.fast_weight_size, self.fast_weight_size)
        )
        self.offset_log_lr = nn.Parameter(
            torch.zeros(self.fast_weight_size, self.fast_weight_size)
        )

        self.gradient_gate_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
        )
        self.offset_gate_proj = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=False,
        )


    def to_lr(self, log_lr: torch.FloatTensor) ->torch.FloatTensor:
        return torch.exp(
            log_lr * self.scalar_scaler
            + math.log(self.base_lr)
            - math.log(self.fast_weight_size)
        )[None]


    def forward(
        self,
        value_prop: torch.FloatTensor,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:

        offset = self.offset_proj(embeddings)

        lr = self.to_lr(self.log_lr)
        offset_lr = self.to_lr(self.offset_log_lr)

        gradient_gate_logits = self.gradient_gate_proj(embeddings)
        offset_gate_logits = self.offset_gate_proj(embeddings)

        return value_prop,offset, lr, offset_lr, gradient_gate_logits, offset_gate_logits


class UnitGLU(nn.Module):
    def forward(self, x, gate):
        return x * F.silu(gate) / 0.6


class ForteFastWeightMLP(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        # save config
        self.intermediate_size = config.intermediate_size
        self.fast_weight_size = config.fast_weight_size

        self.grad_eps = config.grad_rms_eps
        self.offset_alpha = config.offset_alpha

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
        lr_embeddings: torch.FloatTensor | None = None,
        lr_embedding_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:

        mode = ForteMode.INFERENCE
        if fast_weight_mode is not None:
            mode = _tensor_to_mode(fast_weight_mode)

        # fast mlp
        h = self.fast_act_fn(self.up_fast(x), self.gate_fast(x))

        if mode == ForteMode.INFERENCE:
            value = torch.einsum(
                "boi,bli->blo", self.state.detach(), h
            )
            return self.standard_forward(x) + self.down_fast(value)

        if lr_embeddings is None or lr_embedding_mask is None:
            raise ValueError(
                "lr_embeddings and mask are required while training fast weights"
            )

        if mode == ForteMode.TRAIN_FIRST:
            value_prop = torch.einsum(
                "boi,bli->blo", self.state.detach(), h
            )
            output = self.down_fast(value_prop)
            activations_prop = h

        elif mode == ForteMode.TRAIN_SECOND:
            if x.shape[0] % 2 != 0:
                raise ValueError(
                    "train-second inputs and embeddings must have the same "
                    "even batch size"
                )
            if lr_embeddings.shape[0] != x.shape[0] // 2:
                raise ValueError(
                    "train-second lr_embeddings must have half the batch size of inputs"
                )
            activations_replay = maybe_shard_with_gradients(h[::2])
            activations_prop = maybe_shard_with_gradients(h[1::2])

            value_replay = torch.einsum(
                "boi,bli->blo", self.state.detach(), activations_replay
            )
            value_prop = torch.einsum(
                "boi,bli->blo", self.state.detach(), activations_prop
            )
            output = self.down_fast(value_replay)
            output_prop = self.down_fast(value_prop)

        else:
            raise ValueError(f"invalid fast weight mode: {mode}")

        output = ForteFastWeightFunction.apply(
            activations_prop,
            output,
            self.down_fast.weight,
            lr_embedding_mask,
            self.grad_buffer,
            self.state,
            mode,
            {
                "eps": self.grad_eps,
                "offset_alpha": self.offset_alpha,
            },
            *self.fast_dynamic_lr(value_prop, lr_embeddings, lr_embedding_mask),
        )

        if mode == ForteMode.TRAIN_SECOND:
            output = maybe_shard_with_gradients(
                torch.stack([output, output_prop], dim=1).reshape(
                    -1, *output.shape[1:]
                )
            )

        return self.standard_forward(x) + output


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
        mode: ForteMode,
        lr_scale: float = 1.0
    ) -> None:

        G = self.grad_buffer.grad
        update = self.state.grad
        
        self.state.add_(update * lr_scale)

        if mode == ForteMode.TRAIN_FIRST:
            self.grad_buffer.add_(G)
        elif mode == ForteMode.TRAIN_SECOND:
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


class ForteBackboneLayer(LlamaDecoderLayer):
    offload_name = "backbone_decoder_input"

class ForteOutputLayer(LlamaDecoderLayer):
    offload_name = "output_decoder_input"


class ForteModel(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # lm stuff
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # transformer layers
        self.num_backbone_layers = (
            config.num_hidden_layers - config.num_output_layers
        )
        self.backbone_layers = LayerStack(
            config,
            ForteBackboneLayer,
            self.num_backbone_layers,
        )
        self.output_layers = LayerStack(
            config,
            ForteOutputLayer,
            config.num_output_layers,
            layer_offset=self.num_backbone_layers,
        )
        self.lm_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

        # bidirectional head for lr embeddings
        self.embedding_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=False
        )
        
        self.embedding_state_shift = nn.Parameter(
            torch.zeros(config.hidden_size)
        )
        self.embedding_state_scale = nn.Parameter(
            torch.zeros(config.hidden_size)
        )
        self.bidirectional_head = BidirectionalHead(config)

        # fast-weight MLPs
        for layer in self._causal_layers():
            layer.mlp = ForteFastWeightMLP(config)

        # llama stuff
        rope_scaling = config.get("rope_scaling", None)
        if rope_scaling is not None:
            rope_scaling = RopeScaling(**rope_scaling)
        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim=config.hidden_size // config.num_attention_heads,
            rope_theta=config.rope_theta,
            scaling=rope_scaling,
        )

        self.apply(gaussian_init)


    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        **kwargs
    ):
        if any("fast" in k for k in state_dict.keys()):
            return super().load_state_dict(state_dict, **kwargs)

        sd = {}
        for k, v in state_dict.items():

            # model.stuff -> stuff
            if k.startswith("model."):
                k = k.removeprefix("model.")

            # modify layers
            if k.startswith("layers."):
                parts = k.split(".")

                # layers.layers.stuff -> layers.stuff
                if parts[1] == "layers":
                    parts.pop(1)

                # layers.i.stuff -> <type>_layers.layers.i.stuff
                layer_idx = int(parts[1])
                if layer_idx < self.num_backbone_layers:
                    parts = [
                        "backbone_layers",
                        "layers",
                        str(layer_idx)
                    ] + parts[2:]
                else:
                    parts = [
                        "output_layers",
                        "layers",
                        str(layer_idx - self.num_backbone_layers),
                    ] + parts[2:]

                k = ".".join(parts)

            # (model.)norm -> lm_norm
            elif k.startswith("norm."):
                k = "lm_norm." + k.removeprefix("norm.")

            sd[k] = v

        kwargs["strict"] = False
        return super().load_state_dict(
            sd,
            **kwargs
        )


    def _layer_kwargs(
        self,
        hidden_states: torch.FloatTensor,
        embeddings: torch.FloatTensor | None = None,
        embedding_mask: torch.BoolTensor | None = None,
        mode: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        seq_length = hidden_states.shape[1]

        position_ids = torch.arange(
            seq_length, device=hidden_states.device
        ).unsqueeze(0).float()

        kwargs["position_ids"] = position_ids
        kwargs["position_embeddings"] = self.rotary_emb(
            hidden_states, position_ids
        )

        if not (
            self.config.attention_kernel is not None
            and "lash" in self.config.attention_kernel
        ):
            causal_mask = torch.triu(
                torch.full(
                    (seq_length, seq_length),
                    float("-inf"),
                    device=hidden_states.device,
                ),
                diagonal=1,
            )
            kwargs["attention_mask"] = causal_mask[None, None]

        if embeddings is not None:
            kwargs["lr_embeddings"] = embeddings
        if embedding_mask is not None:
            kwargs["lr_embedding_mask"] = embedding_mask.float()
        if mode is not None:
            kwargs["fast_weight_mode"] = _mode_to_tensor(
                mode,
                self.embed_tokens.weight,
            )
        return kwargs


    def forward_backbone(
        self,
        input_ids: torch.LongTensor,
        embeddings: torch.FloatTensor | None = None,
        embedding_mask: torch.BoolTensor | None = None,
        mode: ForteMode | None = None,
    ) -> torch.FloatTensor:

        hidden_states = self.embed_tokens(input_ids)
        kwargs = self._layer_kwargs(
            hidden_states,
            embeddings=embeddings,
            embedding_mask=embedding_mask,
            mode=mode,
        )

        return self.backbone_layers(
            hidden_states,
            **kwargs,
        )


    def forward_lm_states(
        self,
        hidden_states: torch.FloatTensor,
        embeddings: torch.FloatTensor | None = None,
        embedding_mask: torch.BoolTensor | None = None,
        mode: ForteMode | None = None,
        logits_to_keep: slice | None = None,
    ) -> torch.FloatTensor:

        kwargs = self._layer_kwargs(
            hidden_states,
            embeddings=embeddings,
            embedding_mask=embedding_mask,
            mode=mode,
        )

        hidden_states = self.output_layers(
            hidden_states,
            **kwargs,
        )

        if logits_to_keep is not None:
            hidden_states = hidden_states[:, logits_to_keep]

        return self.lm_norm(hidden_states)


    def forward(
        self,
        input_ids: torch.LongTensor,
        embeddings: torch.FloatTensor | None = None,
        embedding_mask: torch.BoolTensor | None = None,
        mode: ForteMode | None = None,
        logits_to_keep: slice | None = None,
    ) -> torch.FloatTensor:
        hidden_states = self.forward_backbone(
            input_ids,
            embeddings=embeddings,
            embedding_mask=embedding_mask,
            mode=mode,
        )
        hidden_states = self.forward_lm_states(
            hidden_states,
            embeddings=embeddings,
            embedding_mask=embedding_mask,
            mode=mode,
            logits_to_keep=logits_to_keep,
        )
        return self.lm_head(hidden_states)


    def forward_embeddings(
        self,
        hidden_states: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = (
            (hidden_states + self.embedding_state_shift)
            * (1.0 + self.embedding_state_scale)
        )
        return self.bidirectional_head(hidden_states, embedding_mask)


    def _causal_layers(self):
        yield from self.backbone_layers._iter_layers()
        yield from self.output_layers._iter_layers()


    def _layer_module(self, layer: LlamaDecoderLayer|int, name: str) -> nn.Module:
        if isinstance(layer, int):
            layer = list(self._causal_layers())[layer]
        try:
            return layer.get_submodule(name)
        except AttributeError:
            return layer._orig_mod.get_submodule(name)


    def fast_modules(self) -> list[ForteFastWeightMLP]:
        return [
            self._layer_module(layer, "mlp")
            for layer in self._causal_layers()
        ]


    def grad_containers(self) -> list[torch.FloatTensor]:
        return [
            mlp.grad_buffer for mlp in self.fast_modules()
        ] + [
            mlp.state for mlp in self.fast_modules()
        ]


    def state_containers(self):
        for mlp in self.fast_modules():
            yield mlp.state


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device) -> None:
        for mlp in self.fast_modules():
            mlp.init_state(bs, device)


    @torch.no_grad()
    def update_state(
        self,
        mode: ForteMode,
        **kwargs
    ) -> None:
        for mlp in self.fast_modules():
            mlp.update_state(mode, **kwargs)


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
        errors = [
            mlp.relative_grad_error() for mlp in self.fast_modules()
        ]
        return torch.stack(errors).mean()
