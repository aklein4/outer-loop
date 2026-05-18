import torch

import torch_xla

from trainers.base_trainer import BaseTrainer
from models.oloop import OLoopModel
from utils.loss_utils import lm_loss_fn
from utils.logging_utils import master_print


class OLoopTrainer(BaseTrainer):
    
    model: OLoopModel


    def post_init(self):
        self.model.init_state(
            self.global_batch_size, self.device
        )


    def get_trainable_parameters(self, model):
        
        slow = []
        fast = []
        embeddings = []
        for name, param in model.named_parameters():

            if "embed_tokens" in name or "lm_head" in name:
                embeddings.append(param)

            elif "log_lr" in name or "fast_out_proj" in name:
                fast.append(param)

            else:
                slow.append(param)

        out = {
            "slow": slow,
            "fast": fast,
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
        )


    @torch_xla.compile(full_graph=True)
    def first_chunk(self, chunk):

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):

            logits = self.model(
                chunk,
                logits_to_keep=slice(0, -1)
            )[0]
            loss = self.loss(chunk, logits)

        loss.backward()

        return loss


    @torch_xla.compile(full_graph=True)
    def looped_chunks(self, in_chunk, out_chunk):

        all_chunk = torch.cat([in_chunk, out_chunk], dim=-1)

        self.model.update_state()

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):

            logits = self.model(
                all_chunk,
                logits_to_keep=slice(in_chunk.shape[-1]-1, -1)
            )[0]
            loss = self.loss(
                all_chunk[:, in_chunk.shape[-1]-1:],
                logits
            )

        loss.backward()

        return loss


    @torch_xla.compile(full_graph=True)
    def post_forward(self):

        # clear state
        self.model.empty_state()

        # regular optimization step
        grad_norm = self.clip_gradients()

        aux = self.optimization_step()
        
        self.model.zero_grad(set_to_none=False)

        return aux, grad_norm


    def train_step(self, batch):

        input_ids: torch.LongTensor = batch["input_ids"]
        chunks = torch.split(
            input_ids, self.config.trainer.chunk_size,
            dim=-1
        )

        # first chunk
        total_loss = self.first_chunk(chunks[0])
        aux = {
            "lm_loss/chunk_00": total_loss,
        }
        torch_xla.sync()
        master_print("Chunk 00 completed.")

        # remaining chunks
        for i in range(1, len(chunks)):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
            
            loss = self.looped_chunks(in_chunk, out_chunk)

            aux[f"lm_loss/chunk_{i:02d}"] = loss
            total_loss = total_loss + loss
            torch_xla.sync()
            master_print(f"Chunk {i:02d} completed.")
        
        post_aux, grad_norm = self.post_forward()
        aux.update(post_aux)

        # finalize outputs
        final_loss = total_loss / len(chunks)
        aux["num_atoms"] = input_ids.numel()

        decades = {}
        for key, value in aux.items():

            if "chunk_" in key:
                if key.endswith("00"):
                    continue

                decade = int(key.split("_")[-1][0])

                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(value)

        for decade, values in decades.items():
            aux[f"grouped_lm_loss/decade_{decade:02d}"] = torch.stack(values).mean()

        return final_loss, aux, grad_norm
    