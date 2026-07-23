import torch
import torch.nn.functional as F
import torch_xla

from models.oloop import OLoopModel
from trainers.base_trainer import BaseTrainer
from utils.logging_utils import master_print
from utils.sharding_utils import maybe_shard_with_gradients


class HorizonTrainer(BaseTrainer):

    model: OLoopModel


    def post_init(self):

        self.model.init_state(
            self.global_batch_size, self.device
        )

        self.do_init = self.config.trainer.do_init

        if not self.model.disable_fast_weights:
            for layer in self.model.model.layers:
                module: torch.nn.Module = self.model._layer_submodule(layer, "mlp.fast")
                for name in ["log_lr", "p_r", "p_l"]:
                    param = module.get_parameter(name)
                    param.no_muon = True


    def get_trainable_parameters(self, model):
        
        slow = []
        fast = []
        embeddings = []
        for name, param in model.named_parameters():

            if "embed_tokens" in name or "lm_head" in name:
                embeddings.append(param)

            elif "fast" in name:
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


    def loss(self, input_ids, assistant_mask, logits):
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = assistant_mask[:, 1:].float().contiguous()

        losses = F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)

        losses = (
            (losses * shift_mask).sum(dim=1) /
            shift_mask.sum(dim=1).clamp_min(1)
        )

        return losses.mean()


    @torch_xla.compile(full_graph=True)
    def inner_fn(self, input_ids, assistant_mask):

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):
            
            logits = self.model(
                input_ids,
                logits_to_keep=slice(0, -1),
            )[0]
            loss = self.loss(input_ids, assistant_mask, logits)

        loss.backward()
        self.model.update_state()

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
        assistant_mask: torch.BoolTensor = batch["assistant_mask"]
        attention_mask: torch.BoolTensor = batch["attention_mask"]

        _, horizon_length, sequence_length = input_ids.shape

        if self.do_init:
            master_print("Initializing fast weights...")
            with torch.no_grad():

                init_input_ids = input_ids[:, :self.config.trainer.init_examples].reshape(
                    -1, sequence_length
                )
                init_mask = attention_mask[:, :self.config.trainer.init_examples].reshape(
                    -1, sequence_length
                )

                init_input_ids = maybe_shard_with_gradients(init_input_ids)
                init_mask = maybe_shard_with_gradients(init_mask)

                self.model.enable_init(init_mask)
                self.model.eval()

                with torch.autocast(
                    "xla",
                    dtype=torch.bfloat16,
                    enabled=self.config.trainer.use_autocast,
                ):
                    self.model(init_input_ids)
                torch_xla.sync(wait=True)

                self.model.disable_init()
                self.model.train()

            self.do_init = False
            master_print("Fast weights initialized.")

        total_loss = 0.0
        aux = {}
        for i in range(horizon_length):

            loss = self.inner_fn(
                input_ids[:, i],
                assistant_mask[:, i]
            )

            # torch_xla.compile schedules execution asynchronously.  Wait for
            # its sharded output placeholders (including the persistent fast
            # weight state) to be populated before using them in another XLA
            # graph.  In particular, this must happen before accumulating the
            # loss below rather than after that graph has already been built.
            torch_xla.sync(wait=True)

            aux[f"lm_loss/episode_{i:02d}"] = loss
            total_loss = total_loss + loss

            master_print(f"Horizon {i:02d} completed.")

        post_aux, grad_norm = self.post_forward()

        # post_forward is compiled too; materialize its optimizer outputs
        # before final_loss and the logging reductions build dependent graphs.
        torch_xla.sync(wait=True)

        aux.update(post_aux)

        # finalize outputs
        final_loss = total_loss / horizon_length
        aux["atom_count"] = attention_mask.long().sum()

        decades = {}
        for key, value in aux.items():

            if "episode_" in key:
                if key.endswith("00"): # skip because it is outlier
                    continue

                decade = int(key.split("_")[-1][0])

                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(value)

        for decade, values in decades.items():
            aux[f"grouped_lm_loss/decade_{decade:02d}"] = torch.stack(values).mean()

        return final_loss, aux, grad_norm
