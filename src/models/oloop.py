import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm
from transformers.activations import ACT2FN

from models.llama import LlamaDecoderLayer, LlamaForCausalLM
from utils.loss_utils import lm_loss_fn
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import fixed_linear, safe_copy_state


def newton_schulz(
    x: torch.FloatTensor,
    steps: int = 6,
    eps: float = 1e-7,
    safety: float = 0.5,
    polar: bool = True,
) -> torch.FloatTensor:
    """Orthogonalize the final two dimensions of ``x``."""

    polar_coeffs = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        (1.875, -1.25, 0.375),
    ]

    y = safety * (
        x.float() /
        (x.float().norm(dim=(-2, -1), keepdim=True) + eps)
    ).to(x.dtype)

    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        y = y.transpose(-2, -1)

    for i in range(steps):

        if polar:
            a, b, c = (
                polar_coeffs[i]
                if i < len(polar_coeffs)
                else polar_coeffs[-1]
            )
        else:
            a, b, c = (3.4445, -4.7750, 2.0315)

        m = y @ y.transpose(-2, -1)
        n = b * m + c * m @ m
        y = a * y + n @ y

    if transpose:
        y = y.transpose(-2, -1)

    return y


def cut_inv_sqrt(
    x: torch.FloatTensor,
    quantile: float,
) -> torch.FloatTensor:
    """Return a quantile-clipped inverse square root of a PSD matrix."""

    with torch.autocast(str(x.device.type), enabled=False):

        x = x.float()

        u, s_0, vh = torch.linalg.svd(x)

        s_sorted = torch.sort(s_0, dim=-1).values
        rank = round(quantile * (s_0.shape[-1] - 1))
        cut = s_sorted[..., rank][..., None]

        s = torch.maximum(s_0, cut)

        return u @ (torch.rsqrt(s)[..., None] * vh)


def precondition(state, lr, p_l, p_r):

    s = p_l[None] @ state @ p_r[None]
    s = lr[None] * s
    s = p_l.T[None] @ s @ p_r.T[None]

    return s


def _masked_statistics(x, mask):
    """Compute source-equivalent token statistics without dynamic indexing."""

    x = x.float()
    mask = mask.to(dtype=x.dtype)[..., None]
    count = mask.sum().clamp_min(1.0)

    mean = (x * mask).sum(dim=(0, 1)) / count
    centered = x - mean

    std = torch.sqrt(
        (centered.square() * mask).sum(dim=(0, 1)) / count
    )
    covariance = torch.einsum(
        "bsi,bsj->ij", centered * mask, centered
    ) / count

    return mean, std, covariance


def _random_orthogonal(size, device):
    """Generate a Haar-distributed orthogonal matrix using torch operations."""

    q, r = torch.linalg.qr(
        torch.randn(size, size, device=device, dtype=torch.float32)
    )
    diagonal = torch.diagonal(r)
    signs = torch.where(
        diagonal < 0,
        -torch.ones_like(diagonal),
        torch.ones_like(diagonal),
    )
    return q * signs[None]


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

        # [b, r, i]
        update = (
            grad.transpose(-2, -1) @ x
        ).to(dtype)
    
        return None, grad, update


class FastWeight(nn.Module):

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
            math.sqrt(self.in_features)
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

        print(s.shape, x.shape, flush=True)
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
        new_whitened = newton_schulz(
            new_momentum
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

        self.eps = config.rms_norm_eps
        self.inv_quantile = config.inv_quantile

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
        self.down_fast = nn.Linear(
            self.fast_weight_size, self.hidden_size, bias=False
        )

        self.register_buffer(
            "in_scale", torch.ones(self.hidden_size), persistent=True
        )
        self.register_buffer(
            "out_scale", torch.ones(self.hidden_size), persistent=True
        )

        self.do_init = False
        self.init_mask = None


    @torch.no_grad()
    def init_input(self, x):
        _, std, _ = _masked_statistics(x, self.init_mask)

        self.in_scale.copy_(1 / (std + self.eps))
        x = x.float() * self.in_scale[None]

        scaled_mean, _, covariance = _masked_statistics(x, self.init_mask)
        whitening = cut_inv_sqrt(covariance, self.inv_quantile)

        up_weight = (
            _random_orthogonal(self.hidden_size, x.device) @ whitening
        )[:self.fast_weight_size]
        self.up_fast.weight.copy_(
            up_weight.to(self.up_fast.weight.dtype)
        )

        gate_weight = (
            _random_orthogonal(self.hidden_size, x.device) @ whitening
        )[:self.fast_weight_size]
        self.gate_fast.weight.copy_(
            gate_weight.to(self.gate_fast.weight.dtype)
        )

        self.up_fast.bias.copy_(
            -fixed_linear(scaled_mean, self.up_fast.weight).to(
                self.up_fast.bias.dtype
            )
        )
        self.gate_fast.bias.copy_(
            -fixed_linear(scaled_mean, self.gate_fast.weight).to(
                self.gate_fast.bias.dtype
            )
        )


    @torch.no_grad()
    def init_output(self, y):
        _, std, _ = _masked_statistics(y, self.init_mask)

        self.out_scale.copy_(std)
        y = y.float() / (self.out_scale[None] + self.eps)

        _, _, covariance = _masked_statistics(y, self.init_mask)

        u, singular_values, vh = torch.linalg.svd(covariance)
        coloring = (
            u @ (torch.sqrt(singular_values + self.eps)[..., None] * vh)
        )

        down_weight = (
            coloring @ _random_orthogonal(self.hidden_size, y.device)
        )[:, :self.fast_weight_size]
        down_weight = down_weight * math.sqrt(
            self.hidden_size / self.fast_weight_size
        )
        self.down_fast.weight.copy_(
            down_weight.to(self.down_fast.weight.dtype)
        )


    def forward(self, x):

        h_base = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        y_base = self.down_proj(h_base)

        if self.do_init:
            self.init_input(x)
            self.init_output(y_base)
            return y_base

        x_fast = x * self.in_scale[None]
        h_fast = self.fast_act_fn(
            self.up_fast(x_fast), self.gate_fast(x_fast)
        )
        h_fast = self.fast(h_fast)
        y_fast = self.down_fast(h_fast)
        y_fast = y_fast * self.out_scale[None]

        return y_base + y_fast


class OLoopModel(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)

        self.disable_fast_weights = config.disable_fast_weights
        if self.disable_fast_weights:
            return

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            mlp = FastWeightMLP(config)
            safe_copy_state(layer.mlp, mlp, strict=False)
            layer.mlp = mlp


    def enable_init(self, mask):
        for module in self.modules():
            if isinstance(module, FastWeightMLP):
                module.do_init = True
                module.init_mask = mask

    def disable_init(self):
        for module in self.modules():
            if isinstance(module, FastWeightMLP):
                module.do_init = False
                module.init_mask = None


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for module in self.modules():
            if isinstance(module, FastWeight):
                module.init_state(bs, device)

    @torch.no_grad()
    def empty_state(self):
        for module in self.modules():
            if isinstance(module, FastWeight):
                module.empty_state()


    @torch.no_grad()
    def update_state(self):
        # stacked FastWeight modules are updated in parallel for efficiency

        to_update = []
        for name, module in self.model.layers[0].named_modules():
            if isinstance(module, FastWeight):
                to_update.append(name)

        for name in to_update:
            self.update_state_named(name)
            

    def _layer_submodule(self, layer, name):
        try:
            return layer.get_submodule(name)
        except AttributeError:
            return layer._orig_mod.get_submodule(name)


    @torch.no_grad()
    def update_state_named(self, name: str):
         # updates named module across all layers in parallel
        
        ref: FastWeight = self._layer_submodule(
            self.model.layers[0], name
        )

        updates = []
        momentums = []
        prev_whiteneds = []
        for layer in self.model.layers:

            module: FastWeight = self._layer_submodule(layer, name)

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
        new_whiteneds = newton_schulz(
            new_momentums
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

        for i, layer in enumerate(self.model.layers):

            module: FastWeight = self._layer_submodule(layer, name)

            module.state.add_(-deltas[:, i])

            module.momentum.copy_(new_momentums[:, i])
            module.momentum.grad.zero_()

            module.prev_whitened.copy_(new_whiteneds[:, i])


    def get_logits(self, *args, **kwargs):
        return self.compute_logits(*args, **kwargs)

    def compute_logits(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor | None = None,
        chunk_size: int | None = None,
        cpu_logits: bool = False,
        verbose: bool = False,
        add_bos: bool = False,
    ):
        if output_ids is not None:
            input_ids = torch.cat([input_ids, output_ids], dim=-1)

        if chunk_size is None:
            chunk_size = self.config.chunk_size

        chunks = torch.split(input_ids, chunk_size, dim=-1)
        ac_kwargs = {
            "device_type": str(input_ids.device),
            "dtype": torch.bfloat16,
        }

        self.init_state(input_ids.shape[0], input_ids.device)
        all_logits = []

        with torch.enable_grad():
            with torch.autocast(**ac_kwargs):
                logits = self(
                    chunks[0], logits_to_keep=slice(0, -1)
                )[0]
                loss = lm_loss_fn(
                    logits,
                    chunks[0],
                    shift_logits=False,
                    ignore_index=self.config.pad_token_id,
                )

                if cpu_logits:
                    logits = logits.cpu()
                all_logits.append(logits.detach())

            loss.backward()

        for i in tqdm(
            range(1, len(chunks)),
            desc="Processing Chunks",
            leave=False,
            disable=not verbose,
        ):
            first_chunk = chunks[i - 1]
            second_chunk = chunks[i]

            if i > 1 and add_bos:
                first_chunk = torch.cat(
                    [
                        torch.full_like(
                            first_chunk[:, :1], self.config.bos_token_id
                        ),
                        first_chunk,
                    ],
                    dim=-1,
                )

            all_chunk = torch.cat([first_chunk, second_chunk], dim=-1)
            self.update_state()

            with torch.enable_grad():
                with torch.autocast(**ac_kwargs):
                    logits = self(
                        all_chunk,
                        logits_to_keep=slice(
                            first_chunk.shape[-1] - 1, -1
                        ),
                    )[0]
                    loss = lm_loss_fn(
                        logits,
                        all_chunk[:, first_chunk.shape[-1]:],
                        shift_logits=False,
                        shift_labels=False,
                        ignore_index=self.config.pad_token_id,
                    )

                    if cpu_logits:
                        logits = logits.cpu()
                    all_logits.append(logits.detach())

                loss.backward()

        self.zero_grad(set_to_none=True)
        self.empty_state()

        logits = torch.cat(all_logits, dim=1).detach()
        if output_ids is not None:
            logits = logits[:, -output_ids.shape[-1]:]

        return logits
