import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
if constants.XLA_AVAILABLE:
    pass

import math
from omegaconf import DictConfig
from tqdm import tqdm

from transformers.activations import ACT2FN

from models.llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import select_newton_schulz
from utils.loss_utils import lm_loss_fn



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
        return y.clone()


    @staticmethod
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        x, = ctx.saved_tensors
        dtype: torch.dtype = ctx.dtype

        # [b, r, i]
        update = (
            grad.to(dtype).transpose(-2, -1) @
            x.to(dtype)
        )
    
        return None, grad, update

        
class FastWeightLoRA(nn.Module):

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
        self.rank = config.fast_weight_rank

        self.base_lr = config.base_lr
        self.momentum_beta = config.momentum_beta

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = getattr(torch, config.momentum_dtype)
        self.state_dtype = getattr(torch, config.state_dtype)
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(self.rank, self.in_features)
        )
        self.out_proj = nn.Linear(
            self.rank, self.out_features, bias=False
        )

        # ephemeral state
        self.state: nn.Buffer
        self.momentum: nn.Buffer
        self.prev_whitened: nn.Buffer

        # weight initialization
        self.out_proj.weight.data.normal_(
            std=1/math.sqrt(self.rank)
        )

    
    @torch.no_grad()
    def svd_init(self, weight):

        u, s, v = torch.linalg.svd(weight, full_matrices=False)

        self.out_proj.weight.copy_(
            u[:, :self.rank]
        )

        self.log_lr.data.add_(
            torch.log(s[:self.rank, None]) / self.scalar_scaler
        )
                    

    def get_lr(self):
        return (
            self.base_lr *
            torch.exp(self.log_lr * self.scalar_scaler) /
            math.sqrt(self.in_features)
        )


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:

        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        s = self.get_lr()[None] * self.state.detach()

        y = torch.einsum("boi,bli->blo", s, x)
        y = FastWeightFunction.apply(x, y, self.momentum)

        y = self.out_proj(y)

        return y


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):

        state = torch.zeros(
            bs, self.rank, self.in_features,
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

        self.momentum.requires_grad_(True)
        self.momentum.grad = torch.zeros_like(self.momentum)
        self.momentum.grad = maybe_shard_with_gradients(self.momentum.grad)

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

        new_momentum = torch.lerp(
            self.momentum,
            update,
            1 - self.momentum_beta
        )
        new_whitened = select_newton_schulz()(
            new_momentum, eps=self.eps
        )

        # approximates newton_schulz as a linear function:
        # f(ax) = af(x), f(x+y) = f(x) + f(y)
        delta = (
            (new_whitened - self.prev_whitened * self.momentum_beta) /
            (1 - self.momentum_beta)
        )

        # scale delta to element-wise scale of 1
        delta = delta * math.sqrt(max(self.in_features, self.rank))
        
        self.state.add_(-delta.to(self.state_dtype))
        
        self.momentum.copy_(new_momentum.detach())
        self.momentum.grad.zero_()

        self.prev_whitened.copy_(new_whitened.detach())


class FastWeightMLP(nn.Module):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.down_fast = FastWeightLoRA(
            self.intermediate_size, self.hidden_size, config
        )


    def forward(self, x):
    
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        y_base = self.down_proj(x)
        y_fast = self.down_fast(x)

        return y_base + y_fast


class OLoopModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp = FastWeightMLP(config)


    @torch.no_grad()
    def load_state_dict(self, state_dict, strict = True, assign = False):
        nn.Module.load_state_dict(self, state_dict, strict, assign)

        # svd init if no fast weights in state dict (loading from pretrained LLM)
        if not any(k.endswith("log_lr") for k in state_dict.keys()):

            for layer in self.model.layers:
                layer: LlamaDecoderLayer

                layer.mlp.down_fast.svd_init(
                    layer.mlp.down_proj.weight.data
                )


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for m in self.modules():
            if isinstance(m, FastWeightLoRA):
                m.init_state(bs, device)


    @torch.no_grad()
    def empty_state(self):
        for m in self.modules():
            if isinstance(m, FastWeightLoRA):
                m.empty_state()
    

    @torch.no_grad()
    def update_state(self):

        to_update = []
        for name, mod in self.model.layers[0].named_modules():
            if isinstance(mod, FastWeightLoRA):
                to_update.append(name)

        for name in to_update:
            self.update_state_named(name)


    @torch.no_grad()
    def update_state_named(self, name: str):
        # updates named module across all layers in parallel
        
        try:
            ref: FastWeightLoRA = self.model.layers[0].get_submodule(name)
        except:
            ref: FastWeightLoRA = self.model.layers[0]._orig_mod.get_submodule(name)

        updates = []
        momentums = []
        prev_whiteneds = []
        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            try:
                m: FastWeightLoRA = layer.get_submodule(name)
            except:
                m: FastWeightLoRA = layer._orig_mod.get_submodule(name)

            updates.append(m.momentum.grad)
            momentums.append(m.momentum)
            prev_whiteneds.append(m.prev_whitened)
        
        updates = torch.stack(updates, dim=1)
        momentums = torch.stack(momentums, dim=1)
        prev_whiteneds = torch.stack(prev_whiteneds, dim=1)

        updates = maybe_shard_with_gradients(updates)
        momentums = maybe_shard_with_gradients(momentums)
        prev_whiteneds = maybe_shard_with_gradients(prev_whiteneds)

        new_momentums = torch.lerp(
            momentums,
            updates,
            1 - ref.momentum_beta
        )
        new_whiteneds = select_newton_schulz()(
            new_momentums, eps=ref.eps
        )

        deltas = (
            (new_whiteneds - prev_whiteneds * ref.momentum_beta) /
            (1 - ref.momentum_beta)
        )

        deltas = deltas * math.sqrt(max(ref.in_features, ref.rank))

        state_deltas = -deltas.to(ref.state_dtype)

        for i, layer in enumerate(self.model.layers):
            layer: LlamaDecoderLayer

            try:
                m: FastWeightLoRA = layer.get_submodule(name)
            except:
                m: FastWeightLoRA = layer._orig_mod.get_submodule(name)

            m.state.add_(state_deltas[:, i].detach())

            m.momentum.copy_(new_momentums[:, i].detach())
            m.momentum.grad.zero_()

            m.prev_whitened.copy_(new_whiteneds[:, i].detach())


    def compute_logits(
        self,
        input_ids: torch.LongTensor,
        verbose: bool = False,
    ):
        """
        Written for iTTT with overlapping chunks.
        
        TODO: update
        """
        
        chunks = torch.split(input_ids, self.config.chunk_size, dim=-1)

        ac_kwargs = {
            "device_type": str(input_ids.device),
            "dtype": torch.bfloat16,
        }

        self.init_state(input_ids.shape[0], input_ids.device)

        all_logits = []

        # first chunk
        with torch.autocast(**ac_kwargs):

            logits = self(
                chunks[0],
                logits_to_keep=slice(0, -1)
            )[0]
            all_logits.append(logits.detach().cpu())

            loss = lm_loss_fn(
                logits, chunks[0],
                shift_logits=False,
                ignore_index=self.config.pad_token_id,
            )

        loss.backward()

        # remaining chunks
        for i in tqdm(range(1, len(chunks)), desc="Processing Chunks", leave=False, disable=(not verbose)):
            
            first_chunk = chunks[i-1]
            second_chunk = chunks[i]
            all_chunk = torch.cat([first_chunk, second_chunk], dim=-1)

            self.update_state()

            with torch.autocast(**ac_kwargs):

                logits = self(
                    all_chunk,
                    logits_to_keep=slice(first_chunk.shape[-1]-1, -1)
                )[0]
                all_logits.append(logits.detach().cpu())

                loss = lm_loss_fn(
                    logits,
                    all_chunk[:, first_chunk.shape[-1]:],
                    shift_logits=False,
                    shift_labels=False,
                    ignore_index=self.config.pad_token_id,
                )

            loss.backward()

        self.zero_grad(True)
        self.empty_state()
            
        return torch.cat(all_logits, dim=1).detach()
