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


class FirstFastWeightFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        state_buffer: torch.FloatTensor,
        eps: float,
    ) -> torch.FloatTensor:
        """
        Args:
            x: input to the fast weights [B, S, I]
            y: output of the fast weights [B, S, O]
            grad_buffer: buffer to store gradients for fast weights [B, O, I]
            state_buffer: buffer to store fast weight state [B, O, I]
            eps: epsilon for numerical stability in Newton-Schulz
        """
        ctx.save_for_backward(x)
        ctx.grad_dtype = grad_buffer.dtype
        ctx.state_dtype = state_buffer.dtype
        ctx.eps = eps
        return y.clone()


    @staticmethod
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        x, = ctx.saved_tensors
        grad_dtype = ctx.grad_dtype
        state_dtype = ctx.state_dtype

        # [b, r, i]
        G = (
            grad.to(torch.bfloat16).transpose(-2, -1) @
            x.to(torch.bfloat16)
        )

        update = -select_newton_schulz()(
            G, steps=3, eps=ctx.eps
        )
    
        return None, grad, G.to(grad_dtype), update.to(state_dtype), None


class SecondFastWeightFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        grad_buffer: torch.FloatTensor,
        state_buffer: torch.FloatTensor,
        eps: float,
        final_grad: torch.FloatTensor,
        out_weight: torch.FloatTensor,
        lr: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            x: input to the fast weights [B, S, I]
            y: output of the *projection after the fast weights* [B, S, O']
            grad_buffer: buffer to store gradients for fast weights [B, O, I]
            state_buffer: buffer to store fast weight state [B, O, I]
            eps: epsilon for numerical stability in Newton-Schulz
            final_grad: the total gradient of the fast weights from the end of the sequence
            out_weight: the weight of the projection after the fast weights [O', O]
            lr: the learning rate for the fast weight update [O, I] 
        """
        ctx.save_for_backward(x, grad_buffer, final_grad, out_weight, lr)
        ctx.grad_dtype = grad_buffer.dtype
        ctx.state_dtype = state_buffer.dtype
        ctx.eps = eps
        return y.clone()


    @staticmethod
    def backward(
        ctx,
        out_grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        x, grad_buffer, final_grad, out_weight = ctx.saved_tensors
        grad_dtype = ctx.grad_dtype
        state_dtype = ctx.state_dtype

        # [b, r, i]
        G = (
            grad.to(torch.bfloat16).transpose(-2, -1) @
            x.to(torch.bfloat16)
        )
        G_future = final_grad - G.to(final_grad.dtype)

        with torch.set_grad_enabled(True):

            x_leaf = x.to(torch.bfloat16).detach().clone().requires_grad_(True)
            g = grad.to(torch.bfloat16)

        update = -select_newton_schulz()(
            G, steps=3, eps=ctx.eps
        )
    
        return None, grad, G.to(grad_dtype), update.to(state_dtype), None

        
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

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = getattr(torch, config.momentum_dtype)
        self.state_dtype = getattr(torch, config.state_dtype)
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(self.out_features, self.in_features)
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


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:

        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        s = self.get_lr()[None] * self.state.detach()

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
        delta = delta * math.sqrt(max(self.in_features, self.out_features))
        
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
        self.fast_weight_size = config.fast_weight_size
        
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.gate_fast = nn.Linear(self.hidden_size, self.fast_weight_size, bias=False)
        self.up_fast = nn.Linear(self.hidden_size, self.fast_weight_size, bias=False)
        self.fast = FastWeight(
            self.fast_weight_size, self.fast_weight_size, config
        )
        self.down_fast = nn.Linear(self.fast_weight_size, self.hidden_size, bias=False)


    def forward(self, x):
    
        h_base = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        y_base = self.down_proj(h_base)

        h_fast = self.act_fn(self.gate_fast(x)) * self.up_fast(x)
        h_fast = self.fast(h_fast)
        y_fast = self.down_fast(h_fast)

        return y_base + y_fast


class OLoopModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        self.disable_fast_weights = config.get("disable_fast_weights", False)
        if self.disable_fast_weights:
            return

        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp = FastWeightMLP(config)


    def load_state_dict(self, state_dict, strict = True, assign = False):

        # svd init if no fast weights in state dict (loading from pretrained LLM)
        if not any(k.endswith("log_lr") for k in state_dict.keys()) and not self.disable_fast_weights:
            nn.Module.load_state_dict(self, state_dict, False, assign)

            with torch.no_grad():

                for layer in self.model.layers:
                    mlp: FastWeightMLP = layer.mlp

                    mlp.gate_fast.weight.data.copy_(
                        mlp.gate_proj.weight.data[:mlp.fast_weight_size, :]
                    )
                    mlp.up_fast.weight.data.copy_(
                        mlp.up_proj.weight.data[:mlp.fast_weight_size, :]
                    )
                    mlp.down_fast.weight.data.copy_(
                        mlp.down_proj.weight.data[:, :mlp.fast_weight_size]
                    )

        else:
            nn.Module.load_state_dict(self, state_dict, strict, assign)


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for m in self.modules():
            if isinstance(m, FastWeight):
                m.init_state(bs, device)


    @torch.no_grad()
    def empty_state(self):
        for m in self.modules():
            if isinstance(m, FastWeight):
                m.empty_state()
    

    @torch.no_grad()
    def update_state(self):

        to_update = []
        for name, mod in self.model.layers[0].named_modules():
            if isinstance(mod, FastWeight):
                to_update.append(name)

        for name in to_update:
            self.update_state_named(name)


    @torch.no_grad()
    def update_state_named(self, name: str):
        # updates named module across all layers in parallel
        
        try:
            ref: FastWeight = self.model.layers[0].get_submodule(name)
        except:
            ref: FastWeight = self.model.layers[0]._orig_mod.get_submodule(name)

        updates = []
        momentums = []
        prev_whiteneds = []
        for layer in self.model.layers:
            layer: LlamaDecoderLayer

            try:
                m: FastWeight = layer.get_submodule(name)
            except:
                m: FastWeight = layer._orig_mod.get_submodule(name)

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

        deltas = deltas * math.sqrt(max(ref.in_features, ref.out_features))

        state_deltas = -deltas.to(ref.state_dtype)

        for i, layer in enumerate(self.model.layers):
            layer: LlamaDecoderLayer

            try:
                m: FastWeight = layer.get_submodule(name)
            except:
                m: FastWeight = layer._orig_mod.get_submodule(name)

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
            all_logits.append(logits.detach())

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
                all_logits.append(logits.detach())

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
