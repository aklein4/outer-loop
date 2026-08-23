import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_leaves, tree_map

from omegaconf import DictConfig
import math

from torchprime.layers.sequential import HomogeneousSequential


def enable_gradient_checkpointing(module: nn.Module, enable: bool = True) -> None:
    def f(m):
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = enable
    module.apply(f)


class LayerStack(nn.Module):

    def __init__(
        self,
        config: DictConfig,
        layer_cls: type[nn.Module],
        num_layers: int,
        layer_offset: int = 0,
    ):
        super().__init__()

        self.layers = HomogeneousSequential(*[
            layer_cls(config, layer_idx=layer_idx+layer_offset)
            for layer_idx in range(num_layers)
        ])

        self.gradient_checkpointing = False


    def __len__(self):
        return len(self.layers)

    def _iter_layers(self):
        for layer in self.layers:
            yield layer


    @staticmethod
    def _tensorize_scalars(carry, kwargs):
        reference = next(
            value for value in tree_leaves(carry)
            if isinstance(value, torch.Tensor)
        )
        dtype = (
            reference.dtype
            if reference.is_floating_point()
            else torch.get_default_dtype()
        )

        def tensorize(value):
            if isinstance(value, (float, int, bool)):
                return torch.tensor(
                    value,
                    device=reference.device,
                    dtype=dtype,
                )
            elif isinstance(value, torch.Tensor) and not value.is_floating_point():
                return value.to(dtype=dtype)
            return value

        return tree_map(tensorize, kwargs)


    def forward(self, *carry, **kwargs):

        kwargs = self._tensorize_scalars(carry, kwargs)

        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            for layer in self._iter_layers():
                output = torch.utils.checkpoint.checkpoint(
                    layer,
                    *carry,
                    use_reentrant=False,
                    **kwargs,
                )
                carry = output if isinstance(output, tuple) else (output,)

            return carry[0] if len(carry) == 1 else carry

        else:
            return self.layers(*carry, **kwargs)


class ScaledEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int
    ):
        super().__init__()

        self.scale = math.sqrt(embedding_dim)

        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim)
            / self.scale
        )


    def forward(self, x):
        return F.embedding(x, self.weight) * self.scale


class ResidualConvMixer(nn.Module):

    no_muon_patterns = [
        "conv"
    ]


    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        init_scale: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.init_scale = init_scale

        self.weight_scale = math.sqrt(self.hidden_size)

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
            bias=False,
        )
        self.conv.weight.data.normal_(
            std=init_scale/(self.weight_scale*math.sqrt(kernel_size))
        )


    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x_mask = x * mask[..., None] if mask is not None else x
        return (
            x +
            self.conv(x_mask.mT).mT * self.weight_scale
        )


class SoftPass(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        pool_size: int,
        output_size: int | None = None,
        do_norm: bool = True,
    ):
        super().__init__()

        self.w_proj = nn.Linear(hidden_size, pool_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, pool_size, bias=True)

        self.r_proj = nn.Linear(pool_size, pool_size, bias=False)
        self.do_norm = do_norm

        self.g_proj = nn.Linear(hidden_size, pool_size, bias=True)

        if output_size is None:
            self.o_proj = nn.Identity()
        else:
            self.o_proj = nn.Linear(pool_size, output_size, bias=False)


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        w = self.w_proj(x)
        if mask is not None:
            w = w.masked_fill(~mask.bool().unsqueeze(-1), -100.0)
        w = F.softmax(w, dim=-2)

        v = self.v_proj(x)
        pooled = (w * v).sum(dim=-2, keepdim=True)
        pooled = self.r_proj(pooled)

        if self.do_norm:
            pooled = F.rms_norm(
                pooled.float(), pooled.shape[-1:], eps=1e-7
            ).to(pooled.dtype)

        g = 2 * torch.sigmoid(self.g_proj(x))
        messages = g * pooled

        return self.o_proj(messages)


class SoftPool(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        pool_size: int,
        output_size: int | None = None,
        do_norm: bool = True,
    ):
        super().__init__()

        self.w_proj = nn.Linear(hidden_size, pool_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, pool_size, bias=False)

        self.do_norm = do_norm

        if output_size is None:
            self.o_proj = nn.Identity()
        else:
            self.o_proj = nn.Linear(pool_size, output_size, bias=False)


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
 
        w = self.w_proj(x)
        if mask is not None:
            w = w.masked_fill(~mask.bool().unsqueeze(-1), -100.0)
        w = F.softmax(w, dim=-2)

        v = self.v_proj(x)
        pooled = (w * v).sum(dim=-2)

        if self.do_norm:
            pooled = F.rms_norm(
                pooled.float(), pooled.shape[-1:], eps=1e-7
            ).to(pooled.dtype)

        return self.o_proj(pooled)
