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

        # gradient DESCENT
        update = -select_newton_schulz()(
            G, eps=ctx.eps
        )
    
        return None, grad, G.to(grad_dtype), update.to(state_dtype), None


class SecondFastWeightFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.FloatTensor,
        out: torch.FloatTensor,
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
            out: output of the *projection after the fast weights* [B, S, P]
            grad_buffer: buffer to store gradients for fast weights [B, O, I]
            state_buffer: buffer to store fast weight state [B, O, I]
            eps: epsilon for numerical stability in Newton-Schulz
            final_grad: the total gradient of the fast weights from the end of the sequence
            out_weight: the weight of the projection after the fast weights [P, O]
            lr: the learning rate for the fast weight update [O, I] 
        """
        ctx.save_for_backward(x, grad_buffer, final_grad, out_weight, lr)
        ctx.grad_dtype = grad_buffer.dtype
        ctx.state_dtype = state_buffer.dtype
        ctx.eps = eps
        return out.clone()


    @staticmethod
    def backward(
        ctx,
        out_grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        x, grad_buffer, final_grad, out_weight, lr = ctx.saved_tensors
        grad_dtype = ctx.grad_dtype
        state_dtype = ctx.state_dtype

        with torch.set_grad_enabled(True):

            x_leaf = x.to(torch.bfloat16).detach().clone().requires_grad_(True)
            out_weight_leaf = out_weight.to(torch.bfloat16).detach().clone().requires_grad_(True)

            grad = torch.einsum(
                "bsp,bpo->bso",
                out_grad.to(torch.bfloat16),
                out_weight_leaf[None]
            )

            # [b, r, i]
            G = (
                grad.transpose(-2, -1) @
                x_leaf
            )
            G_future = (
                final_grad -
                (G.to(final_grad.dtype) + grad_buffer)
            ).detach().to(torch.bfloat16)

            update = -select_newton_schulz()(
                G, eps=ctx.eps
            )

            update_lr = update * lr[None].detach()

            x_grad, out_weight_grad = torch.autograd.grad(
                update_lr,
                (x_leaf, out_weight_leaf),
                grad_outputs=G_future
            )
    
        return (
            x_grad.to(x.dtype),
            out_grad,
            G.to(grad_dtype),
            update.to(state_dtype),
            None,
            None,
            out_weight_grad.to(out_weight.dtype),
            None,
        )

        
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

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.state_dtype = getattr(torch, config.state_dtype)
        
        # params
        self.log_lr = nn.Parameter(
            torch.zeros(self.out_features, self.in_features)
        )
        self.out_proj = nn.Linear(self.out_features, self.out_features, bias=False)

        # ephemeral state
        self.state: nn.Buffer
        self.grad_buffer: nn.Buffer
        self.final_grad_buffer: nn.Buffer

        # first_pass: run the arch like normal
        # second_pass: run the arch with a duplicated mlp forward and estimate the first-order gradients of the state
        self.second_pass = False
                    

    def get_lr(self):
        return (
            self.base_lr *
            torch.exp(self.log_lr * self.scalar_scaler) *
            math.sqrt(max(self.in_features, self.out_features)) /
            math.sqrt(self.in_features)
        )


    def forward(
        self,
        x: torch.FloatTensor,
        x_mlp: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"
        if x_mlp is not None:
            assert x_mlp.shape == x.shape, "x_mlp must have the same shape as x"

        lr = self.get_lr()

        s = lr[None] * self.state.detach()

        y = torch.einsum("boi,bli->blo", s, x)

        if not self.second_pass:
            y = FirstFastWeightFunction.apply(
                x, y,
                self.grad_buffer, self.state,
                self.eps,
            )
            out = self.out_proj(y)

        else:
            assert x_mlp is not None, "x_mlp must be provided for second pass"

            out = self.out_proj(y)
            out = SecondFastWeightFunction.apply(
                x_mlp, out,
                self.grad_buffer, self.state,
                self.eps,
                self.final_grad_buffer,
                self.out_proj.weight, lr,
            )

        return out


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):

        state = torch.zeros(
            bs, self.out_features, self.in_features,
            device=device, dtype=self.state_dtype,
        )
        grad_buffer = torch.zeros_like(
            state, dtype=torch.float32
        )
        final_grad_buffer = torch.zeros_like(
            state, dtype=torch.float32
        )

        state = maybe_shard_with_gradients(state)
        grad_buffer = maybe_shard_with_gradients(grad_buffer)
        final_grad_buffer = maybe_shard_with_gradients(final_grad_buffer)

        self.register_buffer("state", state, persistent=False)
        self.register_buffer("grad_buffer", grad_buffer, persistent=False)
        self.register_buffer("final_grad_buffer", final_grad_buffer, persistent=False)

        self.state.requires_grad_(True)
        self.grad_buffer.requires_grad_(True)
        self.final_grad_buffer.requires_grad_(False)

        self.state.grad = torch.zeros_like(self.state)
        self.state.grad = maybe_shard_with_gradients(self.state.grad)

        self.grad_buffer.grad = torch.zeros_like(self.grad_buffer)
        self.grad_buffer.grad = maybe_shard_with_gradients(self.grad_buffer.grad)


    @torch.no_grad()
    def finalize_gradients(self):
        self.update_state()

        self.state.zero_()
        self.state.grad.zero_()

        self.final_grad_buffer.copy_(self.grad_buffer)
        
        self.grad_buffer.zero_()
        self.grad_buffer.grad.zero_()


    @torch.no_grad()
    def empty_state(self):

        self.state.zero_()
        self.state.grad.zero_()

        self.grad_buffer.zero_()
        self.grad_buffer.grad.zero_()

        self.final_grad_buffer.zero_()

    
    @torch.no_grad()
    def update_state(self):
        
        self.state.add_(self.state.grad)
        self.state.grad.zero_()

        self.grad_buffer.add_(self.grad_buffer.grad)
        self.grad_buffer.grad.zero_()


    @torch.no_grad()
    def relative_grad_error(self):

        est = self.grad_buffer
        target = self.final_grad_buffer

        err = (est - target).norm()
        denom = target.norm() + self.eps

        return err / denom


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
        self.down_fast = FastWeight(
            self.fast_weight_size, self.hidden_size, config
        )


    def forward(self, x):
    
        h_base = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        y_base = self.down_proj(h_base)

        h_fast = self.act_fn(self.gate_fast(x)) * self.up_fast(x)

        h_mlp = None
        if self.down_fast.second_pass:
            x_mlp = x.detach()
            h_mlp = self.act_fn(self.gate_fast(x_mlp)) * self.up_fast(x_mlp)

        y_fast = self.down_fast(h_fast, h_mlp)

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

                    assert mlp.fast_weight_size == mlp.hidden_size, "for svd init, fast_weight_size must be equal to hidden_size"
                    mlp.down_fast.out_proj.weight.data.copy_(
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
    def set_second_pass(self, value):
        for m in self.modules():
            if isinstance(m, FastWeight):
                m.second_pass = value


    @torch.no_grad()
    def finalize_gradients(self):
        for m in self.modules():
            if isinstance(m, FastWeight):
                m.finalize_gradients()


    @torch.no_grad()
    def empty_state(self):
        for m in self.modules():
            if isinstance(m, FastWeight):
                m.empty_state()
    
    
    @torch.no_grad()
    def update_state(self):
        for m in self.modules():
            if isinstance(m, FastWeight):
                m.update_state()

    
    @torch.no_grad()
    def relative_grad_error(self):
        errors = []
        for m in self.modules():
            if isinstance(m, FastWeight):
                errors.append(m.relative_grad_error())
        return torch.stack(errors).mean()


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
