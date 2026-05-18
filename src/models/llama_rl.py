import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

from models.llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaModel, LlamaRMSNorm


class BaseDecoderLayer(LlamaDecoderLayer):
    offload_name = "base_decoder_input"

class BaseLlamaModel(LlamaModel):
    layer_type = BaseDecoderLayer

class BaseLlamaForCausalLM(LlamaForCausalLM):
    model_type = BaseLlamaModel


class LlamaRLModel(nn.Module):
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.base_model = BaseLlamaForCausalLM(config)
        self.rl_model = LlamaForCausalLM(config)

        self.v_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.v_head = nn.Linear(config.hidden_size, 1)
        
        # Squashing V in (-1, 1) is important for binary {-1, +1} rewards
        # We know that V* is in [-1, 1]
        # Enforcing r_+ > V ensures that logp_\theta(y_+|x) > logp_{base}(y_+|x), avoiding the pitfall described in:
        #     - https://arxiv.org/abs/2402.13228
        #     - https://arxiv.org/abs/2412.14516
        # TODO: Similar method for non-binary rewards
        self.v_act = nn.Identity()
        if config.get("squash_v", False):
            self.v_act = nn.Tanh()


    def load_state_dict(self, state_dict, strict = True, assign = False):
            
            # loading from pretrained LLM
            if not any(k.count("rl_model") for k in state_dict.keys()):

                self.base_model.load_state_dict(state_dict, strict=strict, assign=assign)
                self.rl_model.load_state_dict(state_dict, strict=strict, assign=assign)

            else:
                nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)


    def base_forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)[0]
    
    def rl_forward(self, *args, **kwargs):

        kwargs["return_states"] = True
        out_rl = self.rl_model(*args, **kwargs)

        logits_rl = out_rl[0]

        v = self.v_head(self.v_norm(out_rl[-1]))
        v = self.v_act(v).squeeze(-1)

        return logits_rl, v


    def forward(self, *args, **kwargs):

        with torch.no_grad():
            logits_base = self.base_forward(*args, **kwargs).detach()

        logits_rl, v = self.rl_forward(*args, **kwargs)

        return logits_base, logits_rl, v
    