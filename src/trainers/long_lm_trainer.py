import torch

import torch_xla

from models.llama import LlamaForCausalLM
from trainers.base_trainer import BaseTrainer
from utils.loss_utils import lm_loss_fn
from utils.logging_utils import master_print
from utils.sharding_utils import maybe_shard_with_gradients


class LongLMTrainer(BaseTrainer):
    
    model: LlamaForCausalLM


    def get_trainable_parameters(self, model):
        
        normal = []
        embeddings = []
        for name, param in model.named_parameters():

            if "embed_tokens" in name or "lm_head" in name:
                embeddings.append(param)

            else:
                normal.append(param)

        out = {
            "normal": normal,
            "embeddings": embeddings,
        }
        if "embeddings" not in self.config.trainer.multiple_optimizers.keys():
            out.pop("embeddings")
        
        return out


    def loss(self, labels, logits):
        return lm_loss_fn(
            logits, labels,
            ignore_index=self.model.config.pad_token_id,
            shift_logits=False,
            shift_labels=True,
            reduction='none'
        )


    @torch_xla.compile(full_graph=True)
    def grad_accum(self, input_ids):

        input_ids = maybe_shard_with_gradients(input_ids)

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):

            logits = self.model.forward(
                input_ids,
                shift_states=True,
            )[0]
            
            loss = self.loss(input_ids, logits)
            loss_for_backward = loss.mean() / self.config.trainer.grad_accum_steps

        loss_for_backward.backward()

        return loss


    @torch_xla.compile(full_graph=True)
    def post_forward(self, input_ids, losses):

        # regular optimization step
        grad_norm = self.clip_gradients()
        aux = self.optimization_step()
        self.model.zero_grad(set_to_none=False)

        # compute chunk metrics
        losses = torch.cat(losses, dim=0).mean(0)

        chunks = (
            [losses[:self.config.trainer.chunk_size-1]] +
            list(torch.split(losses[self.config.trainer.chunk_size-1:], self.config.trainer.chunk_size, dim=-1))
        )
        for i, chunk in enumerate(chunks):
            aux[f"lm_loss/chunk_{i:02d}"] = chunk.mean()

        # compute decade metrics
        decades = {}
        for key, value in aux.items():

            if "chunk_" in key:
                if key.endswith("00"): # skip because it is outlier
                    continue

                decade = int(key.split("_")[-1][0])

                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(value)

        for decade, values in decades.items():
            aux[f"grouped_lm_loss/decade_{decade:02d}"] = torch.stack(values).mean()

        aux["atom_count"] = input_ids.numel()

        return losses.mean(), aux, grad_norm


    def train_step(self, batch):
        # Assumes the batch has no padding

        input_ids: torch.LongTensor = batch["input_ids"]
        batches = torch.chunk(
            input_ids, self.config.trainer.grad_accum_steps,
            dim=0
        )

        losses = []
        for i, b in enumerate(batches):

            loss = self.grad_accum(b)
            losses.append(loss)

            torch_xla.sync()
            master_print(f"Minibatch {i:02d} completed.")

        return self.post_forward(input_ids, losses)
        