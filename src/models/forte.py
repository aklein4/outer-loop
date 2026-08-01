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
from utils.torch_utils import fixed_linear, gaussian_init

from torchprime.rope.rope import RopeScaling


def _get_G(
    activations: torch.FloatTensor,
    output_grad: torch.FloatTensor,
    down_weight: torch.FloatTensor,
    activation_gate_logits: torch.FloatTensor,
    gradient_gate_logits: torch.FloatTensor,
    eps: float,
) -> torch.FloatTensor:
    # all float to avoid gradient accumulation drift
    a = activations.float()
    g = fixed_linear(
        output_grad.float(), down_weight.T.float()
    )

    # RMS-normalize each hidden channel over the sequence dimension.
    a_norm = a * torch.rsqrt(a.square().mean(dim=-2, keepdim=True) + eps)
    g_norm = g * torch.rsqrt(g.square().sum(dim=-2, keepdim=True) + eps)

    a_gated = 2 * torch.sigmoid(activation_gate_logits.float()) * a_norm
    g_gated = 2 * torch.sigmoid(gradient_gate_logits.float()) * g_norm

    return g.mT @ a, g_gated.mT @ a_gated


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
        activation_gate_logits: torch.FloatTensor,
        gradient_gate_logits: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        state: torch.FloatTensor,
        lr: torch.FloatTensor | None,
        future_loss_scale: torch.FloatTensor,
        grad_eps: float,
        mode: str,
    ) -> torch.FloatTensor:

        to_save = (
            activations,
            down_weight,
            activation_gate_logits,
            gradient_gate_logits,
        )
        if mode == ForteMode.TRAIN_SECOND:
            to_save += (grad_buffer, lr, future_loss_scale)

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

        if mode == ForteMode.TRAIN_FIRST:
            (
                activations,
                down_weight,
                activation_gate_logits,
                gradient_gate_logits,
            ) = ctx.saved_tensors

            G, update = _get_G(
                activations,
                output_grad,
                down_weight,
                activation_gate_logits,
                gradient_gate_logits,
                grad_eps,
            )

            return ( 
                None, # activations
                output_grad, # output
                None, # down_weight
                None, # activation_gate_logits
                None, # gradient_gate_logits
                G, # grad_buffer
                update, # state
                None, # lr
                None, # future_loss_scale
                None, # grad_eps
                None # mode
            )

        elif mode != ForteMode.TRAIN_SECOND:
            raise RuntimeError(f"invalid fast-weight mode in backward: {mode}")

        (
            activations,
            down_weight,
            activation_gate_logits,
            gradient_gate_logits,
            grad_buffer,
            lr,
            future_loss_scale,
        ) = ctx.saved_tensors

        with torch.enable_grad():
            activations_leaf = _get_leaf(activations)
            down_weight_leaf = _get_leaf(down_weight)
            activation_gate_logits_leaf = _get_leaf(activation_gate_logits)
            gradient_gate_logits_leaf = _get_leaf(gradient_gate_logits)
            lr_leaf = _get_leaf(lr)

            G, update = _get_G(
                activations_leaf,
                output_grad,
                down_weight_leaf,
                activation_gate_logits_leaf,
                gradient_gate_logits_leaf,
                grad_eps,
            )

            future_grad = (
                grad_buffer - G
            ).detach()

            with torch.autocast(
                str(future_grad.device.type),
                dtype=torch.bfloat16,
            ):

                state_update = -lr_leaf * update

                local_loss = (
                    future_grad * state_update
                ).sum() * future_loss_scale.detach()

            (
                activation_grad,
                down_weight_grad,
                activation_gate_logits_grad,
                gradient_gate_logits_grad,
                lr_grad,
            ) = torch.autograd.grad(
                local_loss,
                (
                    activations_leaf,
                    down_weight_leaf,
                    activation_gate_logits_leaf,
                    gradient_gate_logits_leaf,
                    lr_leaf,
                ),
            )

        return (
            activation_grad.to(activations.dtype),
            output_grad,
            down_weight_grad.to(down_weight.dtype),
            activation_gate_logits_grad.to(activation_gate_logits.dtype),
            gradient_gate_logits_grad.to(gradient_gate_logits.dtype),
            G, # grad_buffer
            update.detach(), # state
            lr_grad.to(lr.dtype),
            None, # future_loss_scale
            None, # grad_eps
            None, # mode
        )


class DynamicLR(nn.Module):

    no_muon_patterns = (
        "log_lr",
        "activation_gate_proj",
        "gradient_gate_proj",
    )

    def __init__(self, config: DictConfig):
        super().__init__()

        self.fast_weight_size = config.fast_weight_size

        self.base_lr = config.base_lr

        self.scalar_scaler = math.sqrt(self.fast_weight_size)
        self.rms_norm_eps = config.rms_norm_eps

        self.fw_norm = LlamaRMSNorm(
            self.fast_weight_size, eps=self.rms_norm_eps, elementwise_affine=False
        )
        self.hs_norm = LlamaRMSNorm(
            config.hidden_size, eps=self.rms_norm_eps, elementwise_affine=False
        )

        # learning-rate parameters
        num_lr_std_params = 1
        lr_std = config.lr_init_std / math.sqrt(num_lr_std_params)
        self.log_lr = nn.Parameter(
            torch.randn(self.fast_weight_size, self.fast_weight_size)
            * lr_std / self.scalar_scaler
        )

        self.activation_gate_proj = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=False,
        )
        self.gradient_gate_proj = nn.Linear(
            config.hidden_size,
            self.fast_weight_size,
            bias=False,
        )

    def forward(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:

        log_lr = self.log_lr[None] * self.scalar_scaler
 
        return torch.exp(
            log_lr
            + math.log(self.base_lr)
            - math.log(self.fast_weight_size)
        )


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
        future_loss_scale: torch.FloatTensor | float = 1,
    ) -> torch.FloatTensor:

        mode = ForteMode.INFERENCE
        if fast_weight_mode is not None:
            mode = _tensor_to_mode(fast_weight_mode)
        if not isinstance(future_loss_scale, torch.Tensor):
            future_loss_scale = x.new_tensor(future_loss_scale)

        # fast mlp
        h = self.fast_act_fn(self.up_fast(x), self.gate_fast(x))
        value = torch.einsum("boi,bli->blo", self.state.detach(), h)
        output = self.down_fast(value)

        activation_gate_logits = None
        gradient_gate_logits = None
        if mode != ForteMode.INFERENCE:
            activation_gate_logits = (
                self.fast_dynamic_lr.activation_gate_proj(lr_embeddings)
            )
            gradient_gate_logits = (
                self.fast_dynamic_lr.gradient_gate_proj(lr_embeddings)
            )

        activations = h.detach()
        lr = None
        if mode == ForteMode.TRAIN_SECOND:

            x_d = x.detach()
            activations = self.fast_act_fn(
                self.up_fast(x_d), self.gate_fast(x_d)
            )

            lr = self.fast_dynamic_lr(lr_embeddings, lr_embedding_mask)

        if mode != ForteMode.INFERENCE:
            activations = (
                activations
                * lr_embedding_mask[..., None].to(activations.dtype)
            )
            output = ForteFastWeightFunction.apply(
                activations,
                output,
                self.down_fast.weight,
                activation_gate_logits,
                gradient_gate_logits,
                self.grad_buffer,
                self.state,
                lr,
                future_loss_scale,
                self.grad_eps,
                mode,
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
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
        mode: ForteMode,
    ) -> None:

        G = self.grad_buffer.grad
        update = self.state.grad
        
        lr = self.fast_dynamic_lr(embeddings, embedding_mask)
        state_update = -lr * update

        self.state.add_(state_update)

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
        
        self.register_buffer(
            "embedding_state_shift",
            torch.zeros(config.hidden_size),
            persistent=True,
        )
        self.register_buffer(
            "embedding_state_scale",
            torch.ones(config.hidden_size),
            persistent=True,
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
        future_loss_scale: torch.FloatTensor | float | None = None,
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
        if future_loss_scale is not None:
            kwargs["future_loss_scale"] = future_loss_scale

        return kwargs


    def forward_backbone(
        self,
        input_ids: torch.LongTensor,
        embeddings: torch.FloatTensor | None = None,
        embedding_mask: torch.BoolTensor | None = None,
        mode: ForteMode | None = None,
        future_loss_scale: torch.FloatTensor | float | None = None,
    ) -> torch.FloatTensor:

        hidden_states = self.embed_tokens(input_ids)
        kwargs = self._layer_kwargs(
            hidden_states,
            embeddings=embeddings,
            embedding_mask=embedding_mask,
            mode=mode,
            future_loss_scale=future_loss_scale,
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
        future_loss_scale: torch.FloatTensor | float | None = None,
    ) -> torch.FloatTensor:

        kwargs = self._layer_kwargs(
            hidden_states,
            embeddings=embeddings,
            embedding_mask=embedding_mask,
            mode=mode,
            future_loss_scale=future_loss_scale,
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
        future_loss_scale: torch.FloatTensor | float | None = None,
    ) -> torch.FloatTensor:
        hidden_states = self.forward_backbone(
            input_ids,
            embeddings=embeddings,
            embedding_mask=embedding_mask,
            mode=mode,
            future_loss_scale=future_loss_scale,
        )
        hidden_states = self.forward_lm_states(
            hidden_states,
            embeddings=embeddings,
            embedding_mask=embedding_mask,
            mode=mode,
            logits_to_keep=logits_to_keep,
            future_loss_scale=future_loss_scale,
        )
        return self.lm_head(hidden_states)


    def forward_embeddings(
        self,
        hidden_states: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = (
            hidden_states * self.embedding_state_scale
            + self.embedding_state_shift
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


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device) -> None:
        for mlp in self.fast_modules():
            mlp.init_state(bs, device)


    @torch.no_grad()
    def update_state(
        self,
        embeddings: torch.FloatTensor,
        embedding_mask: torch.BoolTensor,
        mode: ForteMode,
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
        errors = [
            mlp.relative_grad_error() for mlp in self.fast_modules()
        ]
        return torch.stack(errors).mean()
