import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla

from collections import defaultdict

from models.slither import SlitherModel
from trainers.base_trainer import BaseTrainer
from utils.logging_utils import master_print
from utils.loss_utils import lm_loss_fn


class SlitherTrainer(BaseTrainer):

    model: SlitherModel


    def post_init(self):

        self.model.init_state(
            self.global_batch_size,
            self.device,
        )

        self.model.embed_tokens.weight.no_muon = True
        try:
            self.model.lm_head.weight.no_muon = True
        except AttributeError:
            self.model.lm_head._orig_mod.weight.no_muon = True
    
        for module in self.model._mechanisms():
            module.read_gate.weight.no_muon = True
            module.odot.no_muon = True
            try:
                module.writer.write_gate.weight.no_muon = True
            except AttributeError:
                module.writer._module.write_gate.weight.no_muon = True


    def _autocast(self):
        return torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        )
    
    
    @torch_xla.compile(full_graph=True)
    def go_forward(
        self,
        input_ids: torch.LongTensor,
        mem_states: torch.FloatTensor | None,
    ):

        with torch.no_grad():

            with self._autocast():
                _, mem_states = self.model.forward(
                    input_ids=input_ids,
                    mem_states=mem_states,
                )

            mem_states = mem_states.float()
            self.model.increment_state(mem_states)

        return mem_states.detach()
        


    @torch_xla.compile(full_graph=True)
    def go_backward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        curr_mem_states: torch.FloatTensor | None,
        curr_mem_grad: torch.FloatTensor | None,
        prev_mem_states: torch.FloatTensor | None,
        total_labels: torch.LongTensor,
    ):

        if curr_mem_states is not None:
            curr_mem_states = curr_mem_states.detach().requires_grad_(True)

            self.model.decrement_state(curr_mem_states)

        if self.config.trainer.decay is not None:
            self.model.scale_state_grad(self.config.trainer.decay)

        if prev_mem_states is not None:
            prev_mem_states = prev_mem_states.detach().requires_grad_(True)

        with self._autocast():

            logits, mem_states = self.model.forward(
                input_ids=input_ids,
                mem_states=prev_mem_states,
            )

            sum_loss = lm_loss_fn(
                logits, labels,
                shift_logits=False, shift_labels=False,
                ignore_index=self.model.config.pad_token_id,
                reduction="sum",
            )
            portion_loss = sum_loss / total_labels.to(sum_loss.dtype)
            chunk_loss = sum_loss / (
                labels != self.model.config.pad_token_id
            ).long().sum().clamp_min(1).to(sum_loss.dtype)

            mem_attn_loss = 0.0
            if curr_mem_grad is not None:
                mem_attn_loss = (mem_states * curr_mem_grad.detach()).sum()

            mem_state_loss = 0.0
            if curr_mem_states is not None:
                mem_state_loss = (mem_states * curr_mem_states.grad.detach()).sum()
  
            loss_for_backward = (
                portion_loss +
                mem_attn_loss +
                mem_state_loss
            )
            
        loss_for_backward.backward()

        prev_mem_grad = (
            prev_mem_states.grad.detach()
            if prev_mem_states is not None
            else None
        )

        return portion_loss.detach(), chunk_loss.detach(), prev_mem_grad


    @torch_xla.compile(full_graph=True)
    def post_forward(self, state_norm):

        res = self.model.get_state_norm()
        err = (
            res / state_norm.clamp_min(self.model.config.rms_norm_eps)
        ).mean()

        self.model.empty_state()

        num_none_grad = len([p for p in self.model.parameters() if p.grad is None])

        grad_norm = self.clip_gradients()
        aux = self.optimization_step()

        self.model.zero_grad(set_to_none=False)

        aux["relative_state_error"] = err
        aux["num_none_grad"] = num_none_grad

        return aux, grad_norm


    def train_step(self, batch):
        tokens: torch.LongTensor = batch["input_ids"]

        input_ids = tokens[:, :-1].split(
            self.model.chunk_length, dim=1
        )
        labels = tokens[:, 1:].split(
            self.model.chunk_length, dim=1
        )
        episodes = list(zip(input_ids, labels))
        total_labels = (
            tokens[:, 1:] != self.model.config.pad_token_id
        ).long().sum().clamp_min(1)

        aux = {}
        mem_stack = []
        portion_losses = []

        assert len(episodes) >= 2

        # first loop
        for index, episode in enumerate(episodes[:-1]):
            input_ids, labels = episode

            mem_states = self.go_forward(
                input_ids=input_ids,
                mem_states=(mem_stack[-1] if len(mem_stack) > 0 else None),
            )

            mem_stack.append(mem_states)

            master_print(
                f"Forward  {index:02d} completed."
            )

        state_norm = self.model.get_state_norm()
  
        # second loop
        curr_mem_grad = None
        for index, episode in list(enumerate(episodes))[::-1]:
            input_ids, labels = episode

            curr_mem_states = mem_stack[index] if index < len(episodes)-1 else None
            prev_mem_states = mem_stack[index-1] if index > 0 else None

            portion_loss, chunk_loss, curr_mem_grad = self.go_backward(
                input_ids=input_ids,
                labels=labels,
                curr_mem_states=curr_mem_states,
                curr_mem_grad=curr_mem_grad,
                prev_mem_states=prev_mem_states,
                total_labels=total_labels,
            )
        
            aux[f"lm_loss/chunk_{index:02d}"] = chunk_loss
            portion_losses.append(portion_loss)

            if curr_mem_states is not None:
                mem_stack.pop()

            master_print(
                f"Backward {index:02d} completed."
            )

        # optimizer step
        post_aux, grad_norm = self.post_forward(state_norm)

        aux.update(post_aux)
        master_print("Optimization step completed.")

        # metrics
        final_loss = torch.stack(portion_losses).sum()
        aux["atom_count"] = total_labels

        decades = defaultdict(list)
        for key, value in aux.items():

            if "chunk_" not in key or key.endswith("00"):
                continue

            decade = int(key.split("_")[-1][0])
            decades[decade].append(value)

        for decade, values in decades.items():
            aux[
                f"grouped_lm_loss/decade_{decade:02d}"
            ] = torch.stack(values).mean()

        return final_loss, aux, grad_norm
