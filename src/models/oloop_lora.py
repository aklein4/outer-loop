import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants

import math
from omegaconf import DictConfig
from tqdm import tqdm

from models.llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import fixed_linear



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

        # [b, o, i]
        update = grad.bfloat16().mT @ x.bfloat16()
    
        return None, grad, update.to(dtype)

        
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
        self.second_moment_beta = config.get("second_moment_beta", math.sqrt(config.momentum_beta))

        self.grad_eps = config.grad_rms_eps
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
        self.second_moment: nn.Buffer
        self.adam_step: nn.Buffer
                    

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
        second_moment = torch.zeros_like(
            state, dtype=self.momentum_dtype
        )
        adam_step = torch.zeros((), device=device, dtype=torch.long)

        state = maybe_shard_with_gradients(state)
        momentum = maybe_shard_with_gradients(momentum)
        second_moment = maybe_shard_with_gradients(second_moment)
    
        self.register_buffer("state", state, persistent=False)
        self.register_buffer("momentum", momentum, persistent=False)
        self.register_buffer("second_moment", second_moment, persistent=False)
        self.register_buffer("adam_step", adam_step, persistent=False)
        
        self.state.requires_grad_(False)

        self.momentum.requires_grad_(True)
        self.momentum.grad = torch.zeros_like(self.momentum)
        self.momentum.grad = maybe_shard_with_gradients(self.momentum.grad)

        self.second_moment.requires_grad_(False)


    @torch.no_grad()
    def empty_state(self):

        self.state.zero_()

        self.momentum.zero_()
        self.momentum.grad.zero_()

        self.second_moment.zero_()

        self.adam_step.zero_()

    
    @torch.no_grad()
    def update_state(self):
        
        update = self.momentum.grad

        new_momentum = torch.lerp(
            self.momentum, update, 1 - self.momentum_beta
        )
        new_second_moment = torch.lerp(
            self.second_moment, update.square(), 1 - self.second_moment_beta
        )
        self.adam_step.add_(1)

        step = self.adam_step.to(new_momentum.dtype)
        first_moment = new_momentum / (1 - self.momentum_beta ** step)
        second_moment = new_second_moment / (1 - self.second_moment_beta ** step)

        delta = first_moment / (second_moment.sqrt() + self.grad_eps)
        
        self.state.add_(-delta.to(self.state_dtype))
        
        self.momentum.copy_(new_momentum.detach())
        self.momentum.grad.zero_()

        self.second_moment.copy_(new_second_moment.detach())


class FastWeightLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        config: DictConfig,
    ):
        super().__init__()
        
        self.weight = base_linear.weight
        self.bias = base_linear.bias

        self.base_down = nn.Linear(
            base_linear.in_features,
            config.fast_weight_rank,
            bias=False,
        )
        self.base_up = nn.Linear(
            config.fast_weight_rank,
            base_linear.out_features,
            bias=False,
        )

        self.fast_down = FastWeight(
            base_linear.in_features,
            config.fast_weight_rank,
            config,
        )
        self.fast_up = FastWeight(
            config.fast_weight_rank,
            base_linear.out_features,
            config,
        )
        self.fast_weights_enabled = True


    def forward(self, x):
    
        y_w = fixed_linear(x, self.weight, self.bias)

        z = self.base_down(x)
        if self.fast_weights_enabled:
            z = z + self.fast_down(x)
            y_fast = self.base_up(z) + self.fast_up(z)
        else:
            y_fast = self.base_up(z)

        return y_w + y_fast


class OLoopLoRAModel(LlamaForCausalLM):


    def __init__(self, config):
        super().__init__(config)

        self.fast_weight_rank = config.fast_weight_rank

        def replace_linear(mod: nn.Module):
            for name, child in mod.named_children():
                if isinstance(child, nn.Linear):
                    mod.set_submodule(
                        name, FastWeightLoRALinear(child, config), True
                    )
                else:
                    replace_linear(child)

        replace_linear(self.model.layers)


    def load_state_dict(self, state_dict, strict = True, assign = False):

        sd = {}
        for k, v in state_dict.items():
            if "layers." in k and "layers.layers." not in k:
                k = k.replace("layers.", "layers.layers.")
            sd[k] = v
        state_dict = sd

        # svd init if no fast weights in state dict (loading from pretrained LLM)
        if not any(k.count("base_down") for k in state_dict.keys()):
            nn.Module.load_state_dict(self, state_dict, False, assign)

            with torch.no_grad():
                
                from tqdm import tqdm
                for mod in tqdm(list(self.modules()), desc="SVD Initialization", leave=False):
                    
                    if isinstance(mod, FastWeightLoRALinear):

                        # svd initialization
                        W = mod.weight.data
                        rank = self.fast_weight_rank

                        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                        U_r = U[:, :rank]
                        S_r = S[:rank]
                        Vh_r = Vh[:rank, :]
                        
                        mod.base_down.weight.data.copy_(torch.diag(torch.sqrt(S_r)) @ Vh_r)
                        mod.base_up.weight.data.copy_(U_r @ torch.diag(torch.sqrt(S_r)))

                        mod.weight.data.sub_(mod.base_up.weight.data @ mod.base_down.weight.data)

        else:
            nn.Module.load_state_dict(self, state_dict, strict, assign)


    def fast_modules(self):
        for module in self.modules():
            if isinstance(module, FastWeight):
                yield module

    def enable_fast_weights_only_for_projection(self, projection: str) -> int:
        """Keep dynamic LoRA weights enabled for only one projection family."""
        count = 0
        for name, module in self.named_modules():
            if isinstance(module, FastWeightLoRALinear):
                enabled = name.endswith(f".{projection}")
                module.fast_weights_enabled = enabled
                count += int(enabled)
        return count

    def _layer_submodule(self, layer: LlamaDecoderLayer|int, name: str) -> nn.Module:
        if isinstance(layer, int):
            layer = list(self.model.layers._iter_layers())[layer]
        try:
            return layer.get_submodule(name)
        except AttributeError:
            return layer._orig_mod.get_submodule(name)

    def _first_layer(self):
        return list(self.model.layers._iter_layers())[0]


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for module in self.fast_modules():
            module.init_state(bs, device)

    @torch.no_grad()
    def empty_state(self):
        for module in self.fast_modules():
            module.empty_state()


    @torch.no_grad()
    def update_state(self):
        # stacked FastWeight modules are updated in parallel for efficiency

        to_update = []
        for name, module in self._first_layer().named_modules():
            if isinstance(module, FastWeight):
                to_update.append(name)

        for name in to_update:
            self.update_state_named(name)


    @torch.no_grad()
    def update_state_named(self, name: str):
        # updates named module across all layers in parallel
        
        ref: FastWeight = self._layer_submodule(0, name)

        updates = []
        momentums = []
        second_moments = []
        for i in range(len(self.model.layers)):

            module: FastWeight = self._layer_submodule(i, name)

            updates.append(module.momentum.grad)
            momentums.append(module.momentum)
            second_moments.append(module.second_moment)
        
        updates = maybe_shard_with_gradients(
            torch.stack(updates, dim=1)
        )
        momentums = maybe_shard_with_gradients(
            torch.stack(momentums, dim=1)
        )
        second_moments = maybe_shard_with_gradients(
            torch.stack(second_moments, dim=1)
        )

        new_momentums = torch.lerp(
            momentums, updates, 1 - ref.momentum_beta
        )
        new_second_moments = torch.lerp(
            second_moments, updates.square(), 1 - ref.second_moment_beta
        )
        ref.adam_step.add_(1)

        step = ref.adam_step.to(new_momentums.dtype)
        first_moments = (
            new_momentums / (1 - ref.momentum_beta ** step)
        )
        corrected_second_moments = (
            new_second_moments / (1 - ref.second_moment_beta ** step)
        )

        deltas = first_moments / (corrected_second_moments.sqrt() + ref.grad_eps)
        state_deltas = -deltas.to(ref.state_dtype)

        for i in range(len(self.model.layers)):
            module: FastWeight = self._layer_submodule(i, name)

            module.state.add_(state_deltas[:, i].detach())

            module.momentum.copy_(new_momentums[:, i].detach())
            module.momentum.grad.zero_()

            module.second_moment.copy_(new_second_moments[:, i].detach())
            module.adam_step.copy_(ref.adam_step)
