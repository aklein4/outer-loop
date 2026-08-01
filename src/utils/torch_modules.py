import torch
import torch.nn as nn
import torch.nn.functional as F

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


    def _iter_layers(self):
        for layer in self.layers:
            yield layer


    def forward(self, carry, **kwargs):

        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            for layer in self._iter_layers():
                carry = torch.utils.checkpoint.checkpoint(
                    layer,
                    carry,
                    use_reentrant=False,
                    **kwargs,
                )

        else:
            carry = self.layers(carry, **kwargs)

        return carry


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
