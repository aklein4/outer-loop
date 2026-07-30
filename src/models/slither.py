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
from utils.attention_utils import AtttentionProbe
from utils.torch_utils import gaussian_init, inv_softplus, unsqueeze_to_batch
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_modules import LayerStack, ScaledEmbedding

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

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.num_heads,
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

        g = 2 * torch.sigmoid(self.gate_proj(hidden_states))
        attn_output = attn_output * g[..., None]

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

        self.num_state_in_heads = config.num_state_in_heads
        self.num_state_out_heads = config.num_state_out_heads

        self.in_head_dim = self.state_size // self.num_state_in_heads

        self.activation = OddActivation()
        self.in_norm = GroupRMSNorm(
            self.state_size, self.num_state_in_heads, eps=config.rms_norm_eps
        )
        self.out_norm = GroupRMSNorm(
            self.state_size, self.num_state_out_heads, eps=config.rms_norm_eps
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

        self.in_gate = nn.Linear(
            config.hidden_size,
            self.num_state_in_heads,
            bias=False,
        )
        self.out_gate = nn.Linear(
            config.hidden_size,
            self.num_state_out_heads,
            bias=False,
        )


    def forward(
        self,
        mem_states: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
        
        key_states = self.activation(self.k_proj(mem_states))
        in_gate = torch.softmax(self.in_gate(mem_states), dim=-1) * self.num_state_in_heads
        key_states = self.in_norm(key_states, scales=in_gate)

        value_states = self.v_proj(mem_states)
        gate = 2 * torch.sigmoid(self.out_gate(mem_states))
        value_states = self.out_norm(value_states, scales=gate)

        update = value_states.mT @ key_states

        k = key_states.view(
            *mem_states.shape[:-1],
            self.num_state_in_heads,
            self.in_head_dim
        ).float()
        corr = torch.einsum(
            "blhd,blhe->bhde", k, k
        )

        count = torch.full_like(
            mem_states[:, 0, 0],
            mem_states.shape[1],
            dtype=torch.int32,
        )

        return update, corr, count


class SlitherStateMechanism(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps

        self.state_size = config.state_size
        self.num_state_in_heads = config.num_state_in_heads
        self.num_state_out_heads = config.num_state_out_heads

        self.in_head_dim = self.state_size // self.num_state_in_heads

        self.mse_solve = config.mse_solve

        self.activation = OddActivation()
        self.in_norm = GroupRMSNorm(
            self.state_size, self.num_state_in_heads, eps=config.rms_norm_eps
        )
        self.out_norm = GroupRMSNorm(
            self.state_size, self.num_state_out_heads, eps=config.rms_norm_eps
        )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.state_size,
            bias=False,
        )

        self.in_gate = nn.Linear(
            self.hidden_size,
            self.num_state_in_heads,
            bias=False,
        )
        self.out_gate = nn.Linear(
            self.hidden_size,
            self.num_state_out_heads,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.state_size,
            self.hidden_size,
            bias=False,
        )

        self.log_out_scale = nn.Parameter(
            torch.tensor([inv_softplus(config.init_state_out_scale)])
            / math.sqrt(self.state_size)
        )
        self.log_lambda = nn.Parameter(
            torch.zeros(self.num_state_in_heads, self.in_head_dim)
        )

        self.odot = torch.nn.Parameter(
            torch.zeros(self.state_size, self.state_size)
        )

        self.writer = SlitherStateWriter(config)

        # ephemeral state
        self.state: nn.Buffer
        self.k_corr: nn.Buffer
        self.k_count: nn.Buffer


    def get_lambda(self):
        return F.softplus(
            self.log_lambda * math.sqrt(self.state_size)
            + inv_softplus(self.config.init_mse_lambda)
        ) + self.eps


    def _solve(self, q: torch.FloatTensor) -> torch.FloatTensor:
        if not self.mse_solve:
            return q

        count = self.k_count.clamp_min(1).to(self.k_corr.dtype)
        corr = self.k_corr / count[:, None, None, None]

        matrix = corr + torch.diag_embed(self.get_lambda())[None]

        rhs = q.view(
            *q.shape[:-1],
            self.num_state_in_heads,
            self.in_head_dim
        )

        with torch.autocast(str(matrix.device.type), enabled=False):
            inverse, _ = torch.linalg.inv_ex(
                matrix.float(),
                check_errors=False,
            )

            solution = torch.einsum(
                "bhoi,blhi->blho",
                inverse,
                rhs.float()
            )

        return solution.reshape(*q.shape).to(q.dtype)


    def get_s(self) -> torch.FloatTensor:

        dot = self.odot + math.sqrt(1.0 / self.state_size)
        s = self.state * dot[None]

        # don't need to divide by count because of the later rms norm

        return s


    def get_out_scale(self) -> torch.FloatTensor:
        return F.softplus(self.log_out_scale * math.sqrt(self.state_size))


    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ):

        query_states = self.activation(self.q_proj(hidden_states))
        in_gate = torch.softmax(self.in_gate(hidden_states), dim=-1) * self.num_state_in_heads
        query_states = self.in_norm(query_states, scales=in_gate)

        query_states = self._solve(query_states)

        s = self.get_s()
        output = torch.einsum("boi,bli->blo", s, query_states)

        out_gate = 2 * torch.sigmoid(self.out_gate(hidden_states))
        output = self.out_norm(output, scales=out_gate)

        return self.o_proj(output) * self.get_out_scale()


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device) -> None:

        state = torch.zeros(
            bs, self.state_size, self.state_size,
            device=device, dtype=torch.float32
        )
        k_corr = torch.zeros(
            bs, self.num_state_in_heads, self.in_head_dim, self.in_head_dim,
            device=device, dtype=torch.float32
        )
        k_count = torch.zeros(
            bs, device=device, dtype=torch.int32
        )

        state = maybe_shard_with_gradients(state)
        k_corr = maybe_shard_with_gradients(k_corr)
        k_count = maybe_shard_with_gradients(k_count)

        self.register_buffer("state", state, persistent=False)
        self.register_buffer("k_corr", k_corr, persistent=False)
        self.register_buffer("k_count", k_count, persistent=False)

        self.state.requires_grad_(True)
        self.state.grad = maybe_shard_with_gradients(
            torch.zeros_like(self.state)
        )

        self.k_corr.requires_grad_(True)
        self.k_corr.grad = maybe_shard_with_gradients(
            torch.zeros_like(self.k_corr)
        )

        # (k_count is not differentiable)


    @torch.no_grad()
    def empty_state(self) -> None:

        self.state.zero_()
        self.state.grad.zero_()

        self.k_corr.zero_()
        self.k_corr.grad.zero_()

        self.k_count.zero_()


    @torch.no_grad()
    def increment_state(self, mem_states: torch.FloatTensor) -> None:
        update, corr, count = self.writer(mem_states)

        self.state.add_(update)
        self.k_corr.add_(corr)
        self.k_count.add_(count)


    def decrement_state(self, mem_states: torch.FloatTensor) -> None:
        update, corr, count = self.writer(mem_states)

        torch.autograd.backward(
            (update, corr),
            (self.state.grad, self.k_corr.grad)
        )

        with torch.no_grad():
            self.state.sub_(update)
            self.k_corr.sub_(corr)
            self.k_count.sub_(count)


    @torch.no_grad()
    def scale_state_grad(self, scale: float) -> None:
        self.state.grad.mul_(scale)
        self.k_corr.grad.mul_(scale)


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

        # State Mechanism (mem_states is a proxy for not-first-chunk)
        if mem_states is not None:
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


class SlitherBackboneLayer(SlitherLayer):
    offload_name = "slither_backbone_layer_input"
    is_causal = True


class SlitherOutputLayer(SlitherLayer):
    offload_name = "slither_output_layer_input"
    is_causal = True

class SlitherMemoryLayer(SlitherLayer):
    offload_name = "slither_memory_layer_input"
    is_causal = False


class SlitherModel(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.state_size = config.state_size
        self.chunk_length = config.chunk_length

        self.scalar_scaler = math.sqrt(config.hidden_size)

        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.embed_tokens = ScaledEmbedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.embed_positions = nn.Parameter(
            torch.randn(self.chunk_length, config.position_embedding_dim) * 0.1
            / self.scalar_scaler
        )
        self.position_proj = nn.Linear(
            config.position_embedding_dim, config.hidden_size, bias=False
        )
        self.embed_first = nn.Parameter(
            torch.randn(config.hidden_size) * 0.1
            / self.scalar_scaler
        )

        # `HomogeneousSequential` is similar to `nn.Sequential` but can be compiled with
        # `scan` described in https://pytorch.org/xla/release/r2.6/features/scan.html.
        self.backbone_layers = LayerStack(
            config,
            SlitherBackboneLayer,
            config.num_hidden_layers,
        )

        self.output_layers = LayerStack(
            config,
            SlitherOutputLayer,
            config.num_output_layers,
            layer_offset=config.num_hidden_layers
        )
        self.lm_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.memory_layers = LayerStack(
            config,
            SlitherMemoryLayer,
            config.num_memory_layers,
            layer_offset=config.num_hidden_layers + config.num_output_layers
        )
        self.mem_norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            elementwise_affine=False
        )

        rope_scaling = config.get("rope_scaling", None)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        if rope_scaling is not None:
            rope_scaling = RopeScaling(**rope_scaling)
        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim=head_dim, rope_theta=config.rope_theta, scaling=rope_scaling
        )

        self.apply(gaussian_init)


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        mem_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None, # only used in non-kernel attention
        position_ids: torch.LongTensor | None = None,
        logits_to_keep: slice | None = None,
        skip_logits: bool = False,
        position_slice: slice | None = None,
    ) -> torch.Tensor:        

        # convert input ids to embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        seq_length = inputs_embeds.shape[1]

        assert seq_length <= self.chunk_length, f"Input sequence length {seq_length} exceeds chunk length {self.chunk_length}"
        if position_slice is None:
            position_slice = slice(0, seq_length)
        inputs_embeds = inputs_embeds + unsqueeze_to_batch(
            self.position_proj(self.embed_positions[position_slice] * self.scalar_scaler),
            inputs_embeds
        )

        # handle mem
        if mem_states is None:
            full_seq_length = seq_length
            inputs_embeds = inputs_embeds + unsqueeze_to_batch(
                self.embed_first * self.scalar_scaler,
                inputs_embeds
            )

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
        
        hidden_states = self.backbone_layers(
            inputs_embeds, **causal_layer_kwargs
        )
        lm_states = self.output_layers(
            hidden_states, **causal_layer_kwargs
        )
        new_mem_states = self.memory_layers(
            hidden_states, **noncausal_layer_kwargs
        )

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

    def _layers(self):
        for layer in self.backbone_layers._iter_layers():
            yield layer
        for layer in self.output_layers._iter_layers():
            yield layer
        for layer in self.memory_layers._iter_layers():
            yield layer

    def _attentions(self):
        for layer in self._layers():
            yield self._layer_module(layer, "self_attn")

    def _mechanisms(self):
        for layer in self._layers():
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
    def scale_state_grad(self, scale: float) -> None:
        for mechanism in self._mechanisms():
            mechanism.scale_state_grad(scale)
            

    @torch.no_grad()
    def get_state_norm(self) -> torch.FloatTensor:
        norms = []
        for mechanism in self._mechanisms():
            norms.append(mechanism.state.norm(dim=(-2, -1)))
        return torch.stack(norms, dim=1)
