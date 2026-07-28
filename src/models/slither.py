import torch
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F

import math
from omegaconf import DictConfig

from torchprime.layers.sequential import HomogeneousSequential
from torchprime.rope.rope import RopeScaling
from torchprime.torch_xla_models.attention import AttentionModule

from utils import constants
if constants.XLA_AVAILABLE:
    from torchprime.torch_xla_models import offloading
else:
    from torch.utils.checkpoint import checkpoint
from utils.attention_utils import AtttentionProbe
from utils.torch_utils import gaussian_init
from utils.sharding_utils import maybe_shard_with_gradients

from models.llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaMLP,
)


class SlitherAttention(nn.Module):

    def __init__(self, config: DictConfig, layer_idx: int, is_causal: bool = True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.is_causal = is_causal
        self.attention_block = AttentionModule(config, is_causal=is_causal)

        self.hidden_size = config.hidden_size
        self.mem_size = config.hidden_size

        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        self.k_proj_mem = nn.Linear(
            self.mem_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj_mem = nn.Linear(
            self.mem_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        self.probe = AtttentionProbe(layer_idx)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        mem_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        bsz, q_len, _ = hidden_states.shape
        has_mem = mem_states is not None

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if has_mem:

            # attn kernel needs square attention, weights
            # discard this later
            mem_query_states = query_states.new_zeros(
                bsz, mem_states.shape[1], query_states.shape[-1]
            )
            
            mem_key_states = self.k_proj_mem(mem_states)
            mem_value_states = self.v_proj_mem(mem_states)

            query_states = torch.cat([mem_query_states, query_states], dim=1)
            key_states = torch.cat([mem_key_states, key_states], dim=1)
            value_states = torch.cat([mem_value_states, value_states], dim=1)

        kv_len = key_states.shape[1]

        query_states = query_states.view(
            bsz, kv_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, kv_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, kv_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = self.attention_block(
            query_states,
            key_states,
            value_states,
            attention_mask,
            attention_probe=self.probe,
        )

        # discard previous padding
        if has_mem:
            attn_output = attn_output[:, :, -q_len:, :]

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class OddActivation(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(x.pow(2))


class GroupRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, num_groups: int, eps: float = 1e-5):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = hidden_size // num_groups
        assert hidden_size % num_groups == 0, f"hidden_size must be divisible by num_groups (got {hidden_size} and {num_groups})"

        self.eps = eps


    def forward(self, x: torch.Tensor, scales: torch.Tensor|None = None) -> torch.Tensor:
        og_shape = x.shape
        assert og_shape[-1] == self.hidden_size, f"Expected input with hidden size {self.hidden_size}, but got {x.shape[-1]}"

        x = x.view(*og_shape[:-1], self.num_groups, self.group_size)
        x = F.rms_norm(x.float(), [self.group_size], eps=self.eps).to(x.dtype)

        if scales is not None:
            x = x * scales[..., None]

        return x.view(*og_shape)


class SlitherStateWriter(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.state_size = config.state_size
        self.num_heads = config.num_state_heads

        self.activation = OddActivation()
        self.rms_norm = LlamaRMSNorm(
            self.state_size, eps=config.rms_norm_eps, elementwise_affine=False
        )
        self.group_norm = GroupRMSNorm(
            self.state_size, self.num_heads, eps=config.rms_norm_eps
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            self.state_size,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.state_size,
            bias=False,
        )
        self.write_gate = nn.Linear(
            config.hidden_size,
            self.num_heads,
            bias=False,
        )


    def forward(
        self,
        mem_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        key_states = self.k_proj(mem_states)
        key_states = self.rms_norm(self.activation(key_states))

        value_states = self.v_proj(mem_states)
        gate = torch.sigmoid(self.write_gate(mem_states))
        value_states = self.group_norm(value_states, scales=gate)

        return value_states.mT @ key_states


class SlitherStateMechanism(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size

        self.state_size = config.state_size
        self.num_heads = config.num_state_heads

        self.activation = OddActivation()
        self.rms_norm = LlamaRMSNorm(
            self.state_size, eps=config.rms_norm_eps, elementwise_affine=False
        )
        self.group_norm = GroupRMSNorm(
            self.state_size, self.num_heads, eps=config.rms_norm_eps
        )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.state_size,
            bias=False,
        )

        self.read_gate = nn.Linear(
            self.hidden_size,
            self.num_heads,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.state_size,
            self.hidden_size,
            bias=False,
        )

        self.odot = torch.nn.Parameter(
            torch.ones(self.state_size, self.state_size)
            / math.sqrt(self.state_size)
        )

        self.writer = SlitherStateWriter(config)

        # ephemeral state
        self.state: nn.Buffer


    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ):

        query_states = self.q_proj(hidden_states)
        query_states = self.rms_norm(self.activation(query_states))

        scaled_state = self.state * self.odot[None]
        output = torch.einsum("boi,bli->blo", scaled_state, query_states)

        gate = torch.sigmoid(self.read_gate(hidden_states))
        output = self.group_norm(output, scales=gate)

        return self.o_proj(output)


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device) -> None:

        state = torch.zeros(
            bs, self.state_size, self.state_size,
            device=device, dtype=torch.float32
        )
        state = maybe_shard_with_gradients(state)

        self.register_buffer("state", state, persistent=False)

        self.state.requires_grad_(True)
        self.state.grad = maybe_shard_with_gradients(
            torch.zeros_like(self.state)
        )


    @torch.no_grad()
    def empty_state(self) -> None:
        self.state.zero_()
        self.state.grad.zero_()



    @torch.no_grad()
    def increment_state(self, mem_states: torch.FloatTensor) -> None:
        self.state.add_(
            self.writer(mem_states)
        )


    def decrement_state(self, mem_states: torch.FloatTensor) -> None:

        update = self.writer(mem_states)
        torch.autograd.backward(
            update,
            self.state.grad
        )

        with torch.no_grad():
            self.state.sub_(update)


class SlitherLayer(nn.Module):

    offload_name = "slither_layer_input"
    is_causal = True
    

    def __init__(self, config: DictConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SlitherAttention(
            config=config, layer_idx=layer_idx, is_causal=self.is_causal
        )
        self.state_mechanism = SlitherStateMechanism(config)
        self.mlp = LlamaMLP(config)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.state_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        mem_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:

        if constants.XLA_AVAILABLE:
            hidden_states = offloading.offload_name(hidden_states, self.offload_name)

        # Self Attention
        attn_states = self.input_layernorm(hidden_states)
        attn_states = self.self_attn(
            hidden_states=attn_states,
            mem_states=mem_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_states

        # State Mechanism
        state_states = self.state_layernorm(hidden_states)
        state_states = self.state_mechanism(
            hidden_states=state_states,
        )
        hidden_states = hidden_states + state_states

        # Fully Connected
        mlp_states = self.post_attention_layernorm(hidden_states)
        mlp_states = self.mlp(mlp_states)
        hidden_states = hidden_states + mlp_states

        return hidden_states


class SlitherCausalLayer(SlitherLayer):
    offload_name = "slither_causal_layer_input"
    is_causal = True

class SlitherNonCausalLayer(SlitherLayer):
    offload_name = "slither_noncausal_layer_input"
    is_causal = False


class SlitherModel(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.state_size = config.state_size
        self.chunk_length = config.chunk_length

        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # `HomogeneousSequential` is similar to `nn.Sequential` but can be compiled with
        # `scan` described in https://pytorch.org/xla/release/r2.6/features/scan.html.
        self.causal_layers = HomogeneousSequential(
            *[
                SlitherCausalLayer(config, layer_idx=layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.lm_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.noncausal_layers = HomogeneousSequential(
            *[
                SlitherNonCausalLayer(
                    config,
                    layer_idx=config.num_hidden_layers+layer_idx,
                )
                for layer_idx in range(config.num_noncausal_layers)
            ]
        )
        self.mem_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=False)

        rope_scaling = config.get("rope_scaling", None)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        if rope_scaling is not None:
            rope_scaling = RopeScaling(**rope_scaling)
        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim=head_dim, rope_theta=config.rope_theta, scaling=rope_scaling
        )

        self.gradient_checkpointing = False

        self.apply(gaussian_init)
        self.embed_tokens.weight.data.normal_(mean=0.0, std=config.initializer_range)


    def gradient_checkpointing_enable(self, enable: bool = True):
        if constants.XLA_AVAILABLE:
            raise NotImplementedError("Gradient checkpointing is not supported on XLA devices")
        self.gradient_checkpointing = enable
    

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        mem_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None, # only used in non-kernel attention
        position_ids: torch.LongTensor | None = None,
        logits_to_keep: slice | None = None,
        skip_logits: bool = False,
    ) -> torch.Tensor:        

        # convert input ids to embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        seq_length = inputs_embeds.shape[1]

        # handle mem
        if mem_states is None:
            full_seq_length = seq_length

        else:
            assert mem_states.ndim == 3, f"Expected mem_states to be rank-3, got {mem_states.ndim}"
            full_seq_length = mem_states.shape[1] + seq_length

        if position_ids is None:
            position_ids = torch.arange(
                full_seq_length, device=inputs_embeds.device
            ).unsqueeze(0)

        # Create a causal attention mask
        if self.config.attention_kernel is not None and "lash" in self.config.attention_kernel:
            assert attention_mask is None, "Custom attention mask not compatible with flash attention"
            causal_mask = None
            noncausal_mask = None

        else:

            causal_mask = torch.triu(
                torch.full((full_seq_length, full_seq_length), float("-inf"), device=inputs_embeds.device),
                diagonal=1,
            )

            # TODO: correct broadcasting for different attention mask shapes
            if attention_mask is not None:
                causal_mask = causal_mask + attention_mask

            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimension

            noncausal_mask = attention_mask
            if noncausal_mask is None:
                noncausal_mask = torch.zeros_like(causal_mask)
            else:
                noncausal_mask = noncausal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimension

        # create position embeddings to be shared across the decoder layers
        # inputs are only used for dtypes
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        # prepare the arguments
        def _no_none(d):
            return {k: v for k, v in d.items() if v is not None}
        layer_kwargs = {
            "position_embeddings": position_embeddings,
            "mem_states": mem_states,
        }
        causal_layer_kwargs = _no_none({**layer_kwargs, "attention_mask": causal_mask})
        noncausal_layer_kwargs = _no_none({**layer_kwargs, "attention_mask": noncausal_mask})
        
        # causal layers
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            
            hidden_states = inputs_embeds
            for layer in self.causal_layers:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    use_reentrant=False,
                    **causal_layer_kwargs,
                )

            new_mem_states = hidden_states
            for layer in self.noncausal_layers:
                new_mem_states = checkpoint(
                    layer,
                    new_mem_states,
                    use_reentrant=False,
                    **noncausal_layer_kwargs,
                )
            
        else:
            hidden_states = self.causal_layers(
                inputs_embeds,
                **causal_layer_kwargs,
            )
            new_mem_states = self.noncausal_layers(
                hidden_states,
                **noncausal_layer_kwargs,
            )

        lm_states = hidden_states
        if logits_to_keep is not None:
            lm_states = lm_states[:, logits_to_keep, :].contiguous()
        lm_states = self.lm_norm(lm_states)

        logits = lm_states
        if not skip_logits:
            logits = self.lm_head(lm_states).float()
        
        new_mem_states = self.mem_norm(new_mem_states)

        return logits, new_mem_states


    def _layer_module(self, layer, name: str) -> nn.Module:
        try:
            return layer.get_submodule(name)
        except AttributeError:
            return layer._orig_mod.get_submodule(name)


    def _mechanisms(self):
        for layer in self.causal_layers:
            yield self._layer_module(layer, "state_mechanism")
        for layer in self.noncausal_layers:
            yield self._layer_module(layer, "state_mechanism")


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device) -> None:
        for mechanism in self._mechanisms():
            mechanism.init_state(bs, device)

    @torch.no_grad()
    def empty_state(self) -> None:
        for mechanism in self._mechanisms():
            mechanism.empty_state()

    @torch.no_grad()
    def increment_state(self, mem_states: torch.FloatTensor) -> None:
        for mechanism in self._mechanisms():
            mechanism.increment_state(mem_states)

    def decrement_state(self, mem_states: torch.FloatTensor) -> None:
        for mechanism in self._mechanisms():
            mechanism.decrement_state(mem_states)


    @torch.no_grad()
    def get_state_norm(self) -> torch.FloatTensor:
        norms = []
        for mechanism in self._mechanisms():
            norms.append(mechanism.state.norm(dim=(-2, -1)))
        return torch.stack(norms, dim=1)
