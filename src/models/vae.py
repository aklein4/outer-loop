"""A matrix-latent VAE built from Forte and Piano pieces.

The encoder runs causal layers followed by a non-causal head. Normalized key
and gated value features write the final states into one matrix latent. The
decoder is a separate causal stack; every decoder MLP reads that same matrix.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers.activations import ACT2FN

from models.layers import BidirectionalHead
from models.llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from torchprime.rope.rope import RopeScaling
from utils.torch_modules import LayerStack
from utils.torch_utils import (
    gaussian_init,
    unsqueeze_to_batch,
    unsqueeze_to_channel,
)
from utils.sharding_utils import maybe_shard_with_gradients


class LatentWriter(nn.Module):
    """Write a matrix from normalized, gated key/value features."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.latent_size = config.latent_size
        self.eps = config.rms_norm_eps

        self.key_proj = nn.Linear(config.hidden_size, self.latent_size, bias=False)
        self.value_proj = nn.Linear(config.hidden_size, self.latent_size, bias=False)

        self.value_gate_proj = nn.Linear(
            config.hidden_size, self.latent_size, bias=False
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        if valid_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "valid_mask must match the batch and sequence dimensions of "
                f"hidden_states, got {tuple(valid_mask.shape)} and "
                f"{tuple(hidden_states.shape[:2])}"
            )

        keys = F.rms_norm(
            self.key_proj(hidden_states).float(),
            (self.latent_size,),
            eps=self.eps,
        )
        values = F.rms_norm(
            self.value_proj(hidden_states).float(),
            (self.latent_size,),
            eps=self.eps,
        )
        value_gate = 2.0 * torch.sigmoid(
            self.value_gate_proj(hidden_states).float()
        )
        values = values * value_gate

        mask = valid_mask[..., None].to(dtype=torch.float32)
        keys_f = keys.float() * mask
        values_f = values.float() * mask

        # Every valid token contributes one full-dimensional outer product.
        cross = torch.einsum("bso,bsi->boi", values_f, keys_f)
        return cross.float()


class UnitGLU(nn.Module):
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x * F.silu(gate) / 0.6


class LatentMLP(nn.Module):
    """Piano-style read from the shared matrix latent."""

    no_muon_patterns = ("odot",)

    def __init__(self, config: DictConfig):
        super().__init__()
        self.latent_size = config.latent_size
        self.latent_act_fn = UnitGLU()

        self.up_latent = nn.Linear(
            config.hidden_size, self.latent_size, bias=False
        )
        self.gate_latent = nn.Linear(
            config.hidden_size, self.latent_size, bias=True
        )
        self.sig_latent = nn.Linear(
            config.hidden_size, self.latent_size, bias=True
        )
        self.down_latent = nn.Linear(
            self.latent_size, config.hidden_size, bias=False
        )

        # The additive parameterization makes the initial elementwise scale
        # exactly 1/sqrt(latent_size), while allowing every reader to adapt it.
        self.odot = nn.Parameter(
            torch.zeros(self.latent_size, self.latent_size)
        )

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        query = self.latent_act_fn(
            self.up_latent(x), self.gate_latent(x)
        )
        scaled_latent = latent * (
            math.sqrt(1.0 / self.latent_size) + self.odot
        )[None]
        value = torch.einsum("boi,bsi->bso", scaled_latent, query)
        gate = 2.0 * torch.sigmoid(self.sig_latent(x).float())
        return self.down_latent(value * gate.to(value.dtype))


class LatentDecoderMLP(nn.Module):
    """A Llama MLP with a separate matrix-latent reader submodule."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.latent_mlp = LatentMLP(config)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        base = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )
        return base + self.latent_mlp(x, latent)


class VAEEncoderCausalLayer(LlamaDecoderLayer):
    offload_name = "vae_encoder_causal_input"


class VAEDecoderLayer(LlamaDecoderLayer):
    offload_name = "vae_decoder_input"

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = super().forward(
            hidden_states,
            latent=latent,
            **kwargs,
        )
        return hidden_states, latent


class VAEModel(nn.Module):
    """Single-matrix VAE with independent encoder and causal decoder stacks."""

    no_muon_patterns = (
        LlamaModel.no_muon_patterns
        + LlamaForCausalLM.no_muon_patterns
    )

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.latent_size = config.latent_size
        if config.num_decoder_layers != config.num_hidden_layers:
            raise ValueError(
                "num_decoder_layers must equal the pretrained backbone depth "
                f"({config.num_hidden_layers}), got {config.num_decoder_layers}"
            )
        if config.num_encoder_causal_layers > config.num_hidden_layers:
            raise ValueError(
                "num_encoder_causal_layers cannot exceed num_hidden_layers"
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_causal_layers = LayerStack(
            config,
            VAEEncoderCausalLayer,
            config.num_encoder_causal_layers,
        )
        self.encoder_noncausal = BidirectionalHead(
            config, num_layers=config.num_noncausal_layers
        )
        self.encoder_bidirectional_norm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.encoder_state_shift = nn.Parameter(
            torch.zeros(config.hidden_size)
        )
        self.encoder_state_scale = nn.Parameter(
            torch.zeros(config.hidden_size)
        )
        self.encoder_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.latent_writer = LatentWriter(config)

        self.decoder_layers = LayerStack(
            config,
            VAEDecoderLayer,
            config.num_decoder_layers,
        )
        for layer in self.decoder_layers._iter_layers():
            layer.mlp = LatentDecoderMLP(config)
        self.decoder_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder_radius_embedding = nn.Parameter(
            torch.randn(config.hidden_size)
        )

        rope_scaling = config.get("rope_scaling", None)
        if rope_scaling is not None:
            rope_scaling = RopeScaling(**rope_scaling)
        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim=config.hidden_size // config.num_attention_heads,
            rope_theta=config.rope_theta,
            scaling=rope_scaling,
        )

        # Trainer-controlled, persistent VAE schedule state.  NaN is the
        # checkpoint-safe sentinel meaning "initialize from the first batch".
        self.log_alpha: nn.Buffer
        self.register_buffer("log_alpha", torch.tensor(float("nan")))

        self.noise_started: nn.Buffer
        self.register_buffer("noise_started", torch.tensor(False))
        self.noise_step: nn.Buffer
        self.register_buffer("noise_step", torch.tensor(0, dtype=torch.long))

        self.apply(gaussian_init)
        self.initialize_decoder_radius_embedding()


    @torch.no_grad()
    def initialize_decoder_radius_embedding(self) -> None:
        """Match each coordinate's vocabulary-embedding variance in expectation."""
        variance = self.embed_tokens.weight.float().var(
            dim=0, unbiased=False
        )
        sample = torch.randn_like(variance) * torch.sqrt(
            variance.clamp_min(self.config.rms_norm_eps)
        )
        self.decoder_radius_embedding.copy_(
            sample.to(self.decoder_radius_embedding.dtype)
        )


    @staticmethod
    def frobenius_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
        return F.rms_norm(
            x.float(), x.shape[-2:], eps=eps
        ).to(dtype=x.dtype)


    def get_alpha(self, log_alpha: torch.Tensor | None = None) -> torch.Tensor:
        if log_alpha is None:
            log_alpha = self.log_alpha
        return torch.sqrt(torch.exp(log_alpha.float()))


    def sample_latent(
        self,
        mu: torch.Tensor,
        radius: torch.Tensor | float,
        noise_scale: torch.Tensor | float = 1.0,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(mu)
        if isinstance(radius, torch.Tensor):
            radius = unsqueeze_to_channel(radius, mu).to(mu.dtype) 

        return (
            radius * mu
            + noise_scale * noise
        )


    def sample_radius(self, reference: torch.Tensor) -> torch.Tensor:
        """Sample one independent signal scale for every sequence."""
        minimum = self.config.get("radius_min", 0.0)
        maximum = self.config.get("radius_max", 1.0)
        if maximum <= minimum:
            raise ValueError("radius_max must exceed radius_min")
        sample = torch.rand_like(reference[:, 0, 0].float())
        return minimum + (maximum - minimum) * sample


    def _causal_kwargs(self, hidden_states: torch.Tensor) -> dict:
        sequence_length = hidden_states.shape[1]
        position_ids = torch.arange(
            sequence_length, device=hidden_states.device
        ).unsqueeze(0).float()
        kwargs = {
            "position_ids": position_ids,
            "position_embeddings": self.rotary_emb(hidden_states, position_ids),
        }
        if not (
            self.config.attention_kernel is not None
            and "lash" in self.config.attention_kernel
        ):
            causal_mask = torch.triu(
                torch.full(
                    (sequence_length, sequence_length),
                    float("-inf"),
                    device=hidden_states.device,
                ),
                diagonal=1,
            )
            kwargs["attention_mask"] = causal_mask[None, None]
        return kwargs


    def forward(
        self,
        input_ids: torch.LongTensor,
        valid_mask: torch.BoolTensor,
        noise_scale: torch.Tensor | float = 1.0,
        alpha: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        radius: torch.Tensor | float | None = None,
        logits_to_keep: slice | None = None,
        skip_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.encode(input_ids, valid_mask, alpha=alpha)
        if radius is None:
            radius = self.sample_radius(mu)
        latent = self.sample_latent(
            mu,
            noise_scale=noise_scale,
            noise=noise,
            radius=radius,
        )
        logits = self.decode(
            input_ids,
            latent,
            radius=radius,
            logits_to_keep=logits_to_keep,
            skip_logits=skip_logits,
        )
        return logits, mu, radius



    def encode(
        self,
        input_ids: torch.LongTensor,
        valid_mask: torch.BoolTensor,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.encoder_causal_layers(
            hidden_states, **self._causal_kwargs(hidden_states)
        )
        hidden_states = self.encoder_bidirectional_norm(hidden_states)
        hidden_states = (
            hidden_states + self.encoder_state_shift
        ) * (1.0 + self.encoder_state_scale)
        hidden_states = self.encoder_noncausal(
            hidden_states, pad_mask=valid_mask
        )
        # The latent writer consumes the unmasked, post-norm encoder state.
        hidden_states = self.encoder_norm(hidden_states)
        pre_norm_mu = self.latent_writer(hidden_states, valid_mask)
        if alpha is None:
            alpha = self.get_alpha()
        mu = alpha.to(pre_norm_mu.dtype) * self.frobenius_rms_norm(
            pre_norm_mu, self.config.rms_norm_eps
        )
        return mu


    def decode(
        self,
        input_ids: torch.LongTensor,
        latent: torch.Tensor,
        radius: torch.Tensor | float | None = None,
        logits_to_keep: slice | None = None,
        skip_logits: bool = False,
    ) -> torch.Tensor:
        if radius is None:
            radius = torch.ones_like(latent[:, 0, 0].float())

        # Stabilize all matrix readers without erasing radius's signal-to-noise
        # meaning: radius is also supplied explicitly through decoder inputs.
        latent = self.frobenius_rms_norm(
            latent,
            self.config.get("latent_rms_norm_eps", self.config.rms_norm_eps),
        )
        hidden_states = self.embed_tokens(input_ids)
        radius_emb = (
            unsqueeze_to_batch(self.decoder_radius_embedding, hidden_states) *
            unsqueeze_to_channel(radius, hidden_states)
        )
        hidden_states = hidden_states + radius_emb
        hidden_states = maybe_shard_with_gradients(hidden_states)
        latent = maybe_shard_with_gradients(latent)
        kwargs = self._causal_kwargs(hidden_states)
        hidden_states, _ = self.decoder_layers(
            hidden_states, latent, **kwargs
        )
        if logits_to_keep is not None:
            hidden_states = hidden_states[:, logits_to_keep]
        hidden_states = self.decoder_norm(hidden_states)
        if skip_logits:
            return hidden_states
        return self.lm_head(hidden_states).float()


    def latent_modules(self):
        for layer in self.decoder_layers._iter_layers():
            yield layer.mlp.latent_mlp

    def decoder_mlps(self):
        yield from (
            layer.mlp for layer in self.decoder_layers._iter_layers()
        )


    def _load_llama_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """Replicate a causal Llama checkpoint into the VAE's three stacks."""
        normalized = {}
        for key, value in state_dict.items():
            key = key.replace("model.layers.", "model.layers.layers.")
            key = key.replace("model.layers.layers.layers.", "model.layers.layers.")
            normalized[key] = value

        mapped = {}
        if "model.embed_tokens.weight" in normalized:
            mapped["embed_tokens.weight"] = normalized["model.embed_tokens.weight"]
        if "lm_head.weight" in normalized:
            mapped["lm_head.weight"] = normalized["lm_head.weight"]

        norm = normalized.get("model.norm.weight")
        if norm is not None:
            mapped["encoder_norm.weight"] = norm
            mapped["decoder_norm.weight"] = norm

        source_layers = []
        prefix = "model.layers.layers."
        for key in normalized:
            if key.startswith(prefix):
                source_layers.append(int(key[len(prefix):].split(".", 1)[0]))
        if not source_layers:
            raise ValueError("pretrained checkpoint contains no causal layers")
        source_count = max(source_layers) + 1
        if self.config.num_encoder_causal_layers > source_count:
            raise ValueError(
                "num_encoder_causal_layers cannot exceed the pretrained "
                f"checkpoint depth ({source_count}), got "
                f"{self.config.num_encoder_causal_layers}"
            )
        if self.config.num_decoder_layers != source_count:
            raise ValueError(
                "num_decoder_layers must equal the pretrained checkpoint "
                f"depth ({source_count}), got {self.config.num_decoder_layers}"
            )

        def copy_layer(target_prefix: str, target_index: int, source_index: int):
            source_prefix = f"{prefix}{source_index}."
            target = f"{target_prefix}.layers.{target_index}."
            for key, value in normalized.items():
                if key.startswith(source_prefix):
                    mapped[target + key.removeprefix(source_prefix)] = value

        for index in range(self.config.num_encoder_causal_layers):
            copy_layer(
                "encoder_causal_layers",
                index,
                index,
            )
        for index in range(self.config.num_noncausal_layers):
            source_index = min(
                self.config.num_encoder_causal_layers + index,
                source_count - 1,
            )
            copy_layer(
                "encoder_noncausal.layers",
                index,
                source_index,
            )
        for index in range(self.config.num_decoder_layers):
            copy_layer("decoder_layers", index, index)
        return mapped


    def load_state_dict(self, state_dict: dict[str, torch.Tensor], **kwargs):
        if any(key.startswith("encoder_causal_layers.") for key in state_dict):
            return super().load_state_dict(state_dict, **kwargs)

        # A plain Llama checkpoint intentionally leaves writer projections,
        # non-causal mixers, and latent-reader projections at their fresh init.
        mapped = self._load_llama_state_dict(state_dict)
        own_state = self.state_dict()
        mapped = {
            key: value
            for key, value in mapped.items()
            if key in own_state and own_state[key].shape == value.shape
        }
        kwargs["strict"] = False
        result = super().load_state_dict(mapped, **kwargs)
        self.initialize_decoder_radius_embedding()
        return result
