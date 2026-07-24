import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchprime.layers.sequential import HomogeneousSequential
from torchprime.rope.rope import RopeScaling
from torchprime.torch_xla_models.attention import AttentionModule

from models.llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from utils import constants

if constants.XLA_AVAILABLE:
    from torchprime.torch_xla_models import offloading


class BidirectionalAttention(nn.Module):
    """Non-causal Llama attention with custom-Llama elementwise padding."""

    def __init__(
        self,
        config: DictConfig,
        layer_idx: int | None = None,
    ):
        super().__init__()

        self.attention_block = AttentionModule(config, is_causal=False)
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        elementwise_pad_mask=None,
    ) -> torch.FloatTensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        if elementwise_pad_mask is not None:
            query_pad, key_pad = elementwise_pad_mask
            query_scale, query_offset = query_pad
            key_scale, key_offset = key_pad

            query_states = (
                query_states
                * query_scale[:, None].to(query_states.dtype)
                + query_offset[:, None].to(query_states.dtype)
            )
            key_states = (
                key_states
                * key_scale[:, None].to(key_states.dtype)
                + key_offset[:, None].to(key_states.dtype)
            )

        attn_output = self.attention_block(
            query_states,
            key_states,
            value_states,
            attention_mask,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size,
            sequence_length,
            self.hidden_size,
        )
        return self.o_proj(attn_output)


class BidirectionalDecoderLayer(nn.Module):
    offload_name = "bidirectional_head_input"

    def __init__(self, config: DictConfig, layer_idx: int):
        super().__init__()

        self.self_attn = BidirectionalAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        elementwise_pad_mask=None,
    ) -> torch.Tensor:
        if constants.XLA_AVAILABLE:
            hidden_states = offloading.offload_name(
                hidden_states,
                self.offload_name,
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            elementwise_pad_mask=elementwise_pad_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class BidirectionalHead(nn.Module):
    """Llama head with full attention and flash-compatible padding masks."""

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.layers = HomogeneousSequential(
            *[
                BidirectionalDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_bidirectional_layers)
            ]
        )
        self.norm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        rope_scaling = config.get("rope_scaling", None)
        if rope_scaling is not None:
            rope_scaling = RopeScaling(**rope_scaling)
        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim=config.hidden_size // config.num_attention_heads,
            rope_theta=config.rope_theta,
            scaling=rope_scaling,
        )

        self._init_elementwise_pad_mask()

    def _init_elementwise_pad_mask(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        first_index = (head_dim // 2) - 1
        second_index = -1

        query_scales = torch.ones(2, head_dim)
        query_scales[:, first_index] = 0.0
        query_scales[:, second_index] = 0.0
        self.register_buffer(
            "query_scales",
            query_scales,
            persistent=True,
        )

        query_offsets = torch.zeros(2, head_dim)
        query_offsets[:, first_index] = 0.5
        query_offsets[:, second_index] = 0.5
        self.register_buffer(
            "query_offsets",
            query_offsets,
            persistent=True,
        )

        key_scales = query_scales.clone()
        self.register_buffer(
            "key_scales",
            key_scales,
            persistent=True,
        )

        key_offsets = torch.zeros(2, head_dim)
        bias = self.config.get("pad_attention_bias_value", -100.0)
        key_offsets[0, first_index] = bias
        key_offsets[0, second_index] = bias
        self.register_buffer(
            "key_offsets",
            key_offsets,
            persistent=True,
        )

    def _elementwise_pad_mask(self, mask: torch.BoolTensor):
        mask = mask.long()
        return (
            (
                F.embedding(mask, self.query_scales),
                F.embedding(mask, self.query_offsets),
            ),
            (
                F.embedding(mask, self.key_scales),
                F.embedding(mask, self.key_offsets),
            ),
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        elementwise_pad_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        batch_size, sequence_length, _ = hidden_states.shape

        if elementwise_pad_mask is None:
            elementwise_pad_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            )
        elif elementwise_pad_mask.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "elementwise_pad_mask must have shape "
                f"{(batch_size, sequence_length)}, got "
                f"{tuple(elementwise_pad_mask.shape)}"
            )

        position_ids = (
            torch.cumsum(elementwise_pad_mask.long(), dim=-1) - 1
        ).float()
        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
        )
        elementwise_pad_mask = self._elementwise_pad_mask(
            elementwise_pad_mask
        )

        attention_mask = None
        if constants.XLA_AVAILABLE:
            # scan currently requires tensor-valued broadcast inputs.
            attention_mask = torch.zeros_like(position_ids)

        hidden_states = self.layers(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            elementwise_pad_mask=elementwise_pad_mask,
        )
        return self.norm(hidden_states)
