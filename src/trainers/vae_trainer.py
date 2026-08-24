"""Single-pass trainer for the matrix-latent VAE."""

import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

from models.vae import VAEModel
from trainers.base_trainer import BaseTrainer
from utils.scheduling_utils import cosine_warmup
from utils.sharding_utils import maybe_shard_with_gradients


class VAETrainer(BaseTrainer):

    model: VAEModel


    def get_trainable_parameters(self, model):
        slow = []
        fast = []
        embeddings = []
        fast_patterns = (
            "latent_writer",
            "encoder_noncausal",
            "encoder_state",
            "encoder_bidirectional_norm",
            "decoder_radius_embedding",
            "latent_mlp",
        )

        for name, parameter in model.named_parameters():
            if any(
                pattern in name
                for pattern in ("embed_tokens", "lm_head")
            ):
                embeddings.append(parameter)
            elif any(pattern in name for pattern in fast_patterns):
                fast.append(parameter)
            else:
                slow.append(parameter)

        groups = {
            "fast": fast,
            "slow": slow,
            "embeddings": embeddings,
        }
        return {
            name: parameters
            for name, parameters in groups.items()
            if name in self.config.trainer.multiple_optimizers
        }


    def post_init(self):
        # Schedule scalars must agree across every data/FSDP shard.  An empty
        # partition spec is an explicit replicated constraint.
        for name in ("log_alpha", "noise_started", "noise_step"):
            setattr(
                self.model,
                name,
                maybe_shard_with_gradients(
                    getattr(self.model, name), spec=()
                ),
            )


    def flatten_episodes(
        self,
        input_ids: torch.LongTensor,
        assistant_mask: torch.BoolTensor,
        attention_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten trajectories without joining or carrying state across episodes."""
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)[:self.config.trainer.real_batch_size]
            assistant_mask = assistant_mask.flatten(0, 1)[:self.config.trainer.real_batch_size]
            attention_mask = attention_mask.flatten(0, 1)[:self.config.trainer.real_batch_size]
        elif input_ids.ndim != 2:
            raise ValueError(
                "expected [trajectory, episode, token] or [episode, token] "
                f"inputs, got shape {tuple(input_ids.shape)}"
            )
        if not (
            input_ids.shape == assistant_mask.shape == attention_mask.shape
        ):
            raise ValueError("input_ids, assistant_mask, and attention_mask must match")
        
        # Flattening merges a sharded and an unsharded dimension.  Reapply the
        # intended leading batch partition instead of relying on propagation.
        return tuple(
            maybe_shard_with_gradients(tensor)
            for tensor in (
                input_ids,
                assistant_mask.bool(),
                attention_mask.bool(),
            )
        )


    @torch.no_grad()
    def _initialize_alpha(
        self,
        assistant_count: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # radius ~ Uniform(radius_min, radius_max), so E[radius^2] is the
        # second raw moment of that distribution.  Calibrate alpha against
        # the expected post-radius KL; the realized KL uses the sampled radii.
        radius_min = self.model.config.get("radius_min", 0.0)
        radius_max = self.model.config.get("radius_max", 1.0)
        expected_radius_squared = (
            radius_min**2 + radius_min * radius_max + radius_max**2
        ) / 3.0
        if expected_radius_squared <= 0.0:
            raise ValueError("expected squared radius must be positive")

        # With covariance fixed to I and Frobenius-RMS(mu / alpha) == 1,
        # E[total KL] = B * latent_size^2 * alpha^2 * E[radius^2] / 2.
        alpha_squared = (
            2.0
            * self.config.trainer.init_kl
            * assistant_count.float()
            / (
                batch_size
                * self.model.latent_size**2
                * expected_radius_squared
            )
        ).clamp_min(self.model.config.rms_norm_eps)

        initial_log_alpha = torch.log(alpha_squared)
        log_alpha = torch.where(
            torch.isfinite(self.model.log_alpha),
            self.model.log_alpha,
            initial_log_alpha,
        )
        log_alpha = maybe_shard_with_gradients(log_alpha, spec=())

        with torch.no_grad():
            self.model.log_alpha.copy_(log_alpha.detach())

        return self.model.get_alpha(log_alpha)


    def _advance_vae_schedule(
        self,
        reconstruction_loss: torch.Tensor,
    ) -> None:
        below_threshold = (
            reconstruction_loss.detach()
            < self.config.trainer.reconstruction_loss_threshold
        )
        with torch.no_grad():
            self.model.noise_started.logical_or_(below_threshold)
            self.model.noise_step.add_(self.model.noise_started.long())
            alpha_ready = self.model.noise_step >= self.config.trainer.noise_warmup_steps
            decrement = (
                below_threshold & alpha_ready
            ).to(self.model.log_alpha.dtype)
            self.model.log_alpha.sub_(
                decrement * self.config.trainer.log_alpha_decrement
            )


    @torch_xla.compile(full_graph=False)
    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict, torch.Tensor]:

        with torch.autocast('xla', dtype=torch.bfloat16, enabled=self.config.trainer.use_autocast):
            loss, aux, lm_states, lm_grad = self.forward_with_lm_grad(**batch)

        torch.autograd.backward(lm_states, lm_grad)

        grad_norm = self.clip_gradients()

        aux.update(self.optimization_step())

        self.model.zero_grad(set_to_none=False)

        return loss, aux, grad_norm


    def loss_and_lm_grad(
        self,
        lm_states: torch.Tensor,
        labels: torch.LongTensor,
        target_mask: torch.BoolTensor,
        denominator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backpropagate the LM head in bounded-logit chunks."""
        num_iter = self.config.trainer.num_logit_iterations
        token_count = labels.numel()
        if token_count % num_iter != 0:
            raise ValueError(
                f"{token_count} prediction tokens must be divisible by "
                f"num_logit_iterations={num_iter}"
            )

        lm_states_leaf = maybe_shard_with_gradients(
            lm_states.detach().reshape(
                -1, num_iter, lm_states.shape[-1]
            )
        ).detach().requires_grad_(True)
        labels = maybe_shard_with_gradients(
            labels.reshape(-1, num_iter).contiguous()
        )
        weights = maybe_shard_with_gradients(
            target_mask.reshape(-1, num_iter).to(torch.float32)
            / denominator
        )

        losses = []
        dummy = torch.zeros_like(weights.flatten()[0])
        for index in range(num_iter):
            logits = maybe_shard_with_gradients(
                self.model.lm_head(lm_states_leaf[:, index]).float() + dummy
            )
            token_nll = F.cross_entropy(
                logits,
                labels[:, index].contiguous(),
                reduction="none",
            )
            chunk_loss = (token_nll * weights[:, index]).sum()
            dummy = dummy + chunk_loss.detach()*0  # Ensure dummy is used to avoid in-place operation issues
            chunk_loss.backward()
            xm.optimization_barrier_([dummy])
            losses.append(chunk_loss.detach())

        reconstruction_loss = torch.stack(losses).sum()
        lm_grad = lm_states_leaf.grad.reshape_as(lm_states).to(lm_states.dtype)
        return reconstruction_loss, lm_grad.detach()


    def forward_with_lm_grad(self, input_ids, assistant_mask, attention_mask):
        input_ids, assistant_mask, valid_mask = self.flatten_episodes(
            input_ids, assistant_mask, attention_mask
        )
        target_mask = assistant_mask[:, 1:] & valid_mask[:, 1:]
        assistant_count = target_mask.long().sum()
        denominator = assistant_count.clamp_min(1).float()

        alpha = self._initialize_alpha(
            assistant_count, input_ids.shape[0]
        )
        noise_scale = cosine_warmup(
            self.model.noise_step.float(),
            self.config.trainer.noise_warmup_steps,
        )

        lm_states, mu, radius = self.model(
            input_ids=input_ids,
            valid_mask=valid_mask,
            noise_scale=noise_scale,
            alpha=alpha,
            logits_to_keep=slice(0, -1),
            skip_logits=True,
        )
        reconstruction_loss, lm_grad = self.loss_and_lm_grad(
            lm_states,
            input_ids[:, 1:],
            target_mask,
            denominator,
        )

        sequence_kl = (
            0.5
            * self.model.latent_size**2
            * alpha.square()
            * radius.float().square()
        )
        kl_per_assistant_token = sequence_kl.sum() / denominator
        loss = reconstruction_loss + kl_per_assistant_token
        self._advance_vae_schedule(reconstruction_loss)

        aux = {
            "reconstruction_loss": reconstruction_loss,

            "kl": sequence_kl.mean(),
            "kl_per_assistant_token": kl_per_assistant_token,

            "elbo": loss,

            "alpha": alpha,
            "log_alpha": self.model.log_alpha.clone(),

            "noise_scale": noise_scale,
            "noise_started": self.model.noise_started.clone(),
            "noise_step": self.model.noise_step.clone(),

            "assistant_token_count": assistant_count,
            "atom_count": valid_mask.long().sum(),
        }
        return loss, aux, lm_states, lm_grad
