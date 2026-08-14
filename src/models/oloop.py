import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm
from transformers.activations import ACT2FN

from models.llama import LlamaDecoderLayer, LlamaForCausalLM
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import safe_copy_state, select_newton_schulz


def precondition(state, lr, p_l, p_r):
    return lr[None] * state
    p_l, p_r = p_l[None], p_r[None]
    s = p_l @ state @ p_r
    s = lr[None] * s
    s = p_l.mT @ s @ p_r.mT
    return s


class FastWeightFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        buffer: torch.FloatTensor,
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x)
        ctx.dtype = buffer.dtype
        return y


    @staticmethod
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        x, = ctx.saved_tensors
        dtype: torch.dtype = ctx.dtype

        # [b, o, i]
        update = grad.bfloat16().mT @ x.bfloat16()
    
        return None, grad, update.to(dtype)


class FastWeight(nn.Module):

    no_muon_patterns = (
        "log_lr",
    )

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: DictConfig,
    ):
        super().__init__()

        # save config
        self.in_features = in_features
        self.out_features = out_features

        self.base_lr = config.base_lr
        self.momentum_beta = config.momentum_beta

        self.eps = config.grad_rms_eps
        self.grad_eps = config.grad_rms_eps

        self.momentum_dtype = getattr(torch, config.momentum_dtype)
        self.state_dtype = getattr(torch, config.state_dtype)

        self.scalar_scaler = math.sqrt(self.in_features)

        # ittt params
        self.log_lr = nn.Parameter(
            torch.randn(self.out_features, self.in_features) /
            (2 * self.scalar_scaler)
        )
        self.p_r = nn.Parameter(
            torch.eye(self.in_features)
        )
        self.p_l = nn.Parameter(
            torch.eye(self.out_features)
        )

        # ephemeral state
        self.state: nn.Buffer
        self.momentum: nn.Buffer
        self.prev_whitened: nn.Buffer


    def get_lr(self):
        return (
            self.base_lr *
            torch.exp(self.log_lr * self.scalar_scaler) /
            self.in_features
        )

    def get_s(self):
        return precondition(
            self.state.detach(), self.get_lr(), self.p_l, self.p_r
        )


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        s = self.get_s()

        y = torch.einsum("boi,bli->blo", s, x)
        y = FastWeightFunction.apply(x, y, self.momentum)

        return y
    

    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):

        state = torch.zeros(
            bs, self.out_features, self.in_features,
            device=device, dtype=self.state_dtype,
        )
        momentum = torch.zeros_like(
            state, dtype=self.momentum_dtype
        )
        prev_whitened = torch.zeros_like(
            state, dtype=self.momentum_dtype
        )

        state = maybe_shard_with_gradients(state)
        momentum = maybe_shard_with_gradients(momentum)
        prev_whitened = maybe_shard_with_gradients(prev_whitened)

        self.register_buffer("state", state, persistent=False)
        self.register_buffer("momentum", momentum, persistent=False)
        self.register_buffer("prev_whitened", prev_whitened, persistent=False)

        self.state.requires_grad_(False)

        # FastWeightFunction stores new gradients in momentum.grad
        self.momentum.requires_grad_(True)
        self.momentum.grad = maybe_shard_with_gradients(
            torch.zeros_like(self.momentum)
        )

        self.prev_whitened.requires_grad_(False)


    @torch.no_grad()
    def empty_state(self):

        self.state.zero_()

        self.momentum.zero_()
        self.momentum.grad.zero_()

        self.prev_whitened.zero_()


    @torch.no_grad()
    def update_state(self):

        update = self.momentum.grad

        update = F.rms_norm(
            update.float(),
            update.shape[-2:],
            eps=self.grad_eps,
        ).to(update.dtype)

        new_momentum = (
            update +
            self.momentum_beta * self.momentum
        )
        new_whitened = select_newton_schulz()(
            new_momentum, steps=6, eps=self.eps, polar=True
        )

        prev_whitened_normed = self.prev_whitened * (
            torch.norm(self.momentum, dim=(-2, -1), keepdim=True) /
            (torch.norm(self.prev_whitened, dim=(-2, -1), keepdim=True) + self.eps)
        )
        new_whitened_normed = new_whitened * (
            torch.norm(new_momentum, dim=(-2, -1), keepdim=True) /
            (torch.norm(new_whitened, dim=(-2, -1), keepdim=True) + self.eps)
        )

        delta = (
            new_whitened_normed -
            self.momentum_beta * prev_whitened_normed
        )

        self.state.add_(-delta.to(self.state.dtype))

        self.momentum.copy_(new_momentum.to(self.momentum.dtype))
        self.momentum.grad.zero_()

        self.prev_whitened.copy_(new_whitened.to(self.prev_whitened.dtype))


class UnitGLU(nn.Module):

    def forward(self, x, gate):
        return x * F.silu(gate) / 0.6


class FastWeightMLP(nn.Module):

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fast_weight_size = config.fast_weight_size

        self.act_fn = ACT2FN[config.hidden_act]
        self.fast_act_fn = UnitGLU()

        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

        self.up_fast = nn.Linear(
            self.hidden_size, self.fast_weight_size, bias=True
        )
        self.gate_fast = nn.Linear(
            self.hidden_size, self.fast_weight_size, bias=True
        )
        self.fast = FastWeight(
            self.fast_weight_size, self.fast_weight_size, config
        )
        self.sig_fast = nn.Linear(
            self.hidden_size, self.fast_weight_size, bias=True
        )
        self.down_fast = nn.Linear(
            self.fast_weight_size, self.hidden_size, bias=False
        )


    def standard_forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )


    def forward(self, x):

        q = self.fast_act_fn(
            self.up_fast(x), self.gate_fast(x)
        )
        v = self.fast(q)
        v_gate = v * 2 * torch.sigmoid(self.sig_fast(x))
        y = self.down_fast(v_gate)

        return self.standard_forward(x) + y


class OLoopModel(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)

        for layer in self.model.layers._iter_layers():
            layer: LlamaDecoderLayer

            mlp = FastWeightMLP(config)
            safe_copy_state(layer.mlp, mlp, strict=False)
            layer.mlp = mlp


    def fast_modules(self):
        for module in self.modules():
            if isinstance(module, FastWeight):
                yield module

    def _layer_submodule(self, layer: LlamaDecoderLayer|int, name: str) -> nn.Module:
        if isinstance(layer, int):
            layer = list(self.model.layers._iter_layers())[layer]
        try:
            return layer.get_submodule(name)
        except AttributeError:
            return layer._orig_mod.get_submodule(name)

    def _first_layer(self):
        return list(self.model.layers._iter_layers())[0]


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for module in self.fast_modules():
            module.init_state(bs, device)

    @torch.no_grad()
    def empty_state(self):
        for module in self.fast_modules():
            module.empty_state()


    @torch.no_grad()
    def update_state(self):
        # stacked FastWeight modules are updated in parallel for efficiency

        to_update = []
        for name, module in self._first_layer().named_modules():
            if isinstance(module, FastWeight):
                to_update.append(name)

        for name in to_update:
            self.update_state_named(name)
            

    @torch.no_grad()
    def update_state_named(self, name: str):
         # updates named module across all layers in parallel
        
        ref: FastWeight = self._layer_submodule(0, name)

        updates = []
        momentums = []
        prev_whiteneds = []
        for i in range(len(self.model.layers)):

            module: FastWeight = self._layer_submodule(i, name)

            updates.append(module.momentum.grad)
            momentums.append(module.momentum)
            prev_whiteneds.append(module.prev_whitened)

        updates = maybe_shard_with_gradients(
            torch.stack(updates, dim=1)
        )
        momentums = maybe_shard_with_gradients(
            torch.stack(momentums, dim=1)
        )
        prev_whiteneds = maybe_shard_with_gradients(
            torch.stack(prev_whiteneds, dim=1)
        )

        updates = F.rms_norm(
            updates.float(), updates.shape[-2:], eps=ref.grad_eps
        ).to(updates.dtype)

        new_momentums = (
            updates +
            ref.momentum_beta * momentums
        )
        new_whiteneds = select_newton_schulz()(
            new_momentums, steps=6, eps=ref.eps, polar=True
        )

        prev_whiteneds_normed = prev_whiteneds * (
            torch.norm(momentums, dim=(-2, -1), keepdim=True) /
            (torch.norm(prev_whiteneds, dim=(-2, -1), keepdim=True) + ref.eps)
        )
        new_whiteneds_normed = new_whiteneds * (
            torch.norm(new_momentums, dim=(-2, -1), keepdim=True) /
            (torch.norm(new_whiteneds, dim=(-2, -1), keepdim=True) + ref.eps)
        )

        deltas = (
            new_whiteneds_normed -
            ref.momentum_beta * prev_whiteneds_normed
        )

        deltas = deltas.to(ref.state.dtype)
        new_momentums = new_momentums.to(ref.momentum.dtype)
        new_whiteneds = new_whiteneds.to(ref.prev_whitened.dtype)

        for i in range(len(self.model.layers)):
            module: FastWeight = self._layer_submodule(i, name)

            module.state.add_(-deltas[:, i])

            module.momentum.copy_(new_momentums[:, i])
            module.momentum.grad.zero_()

            module.prev_whitened.copy_(new_whiteneds[:, i])
