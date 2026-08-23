"""Initialize a matrix-latent VAE from Llama and activation statistics."""

import argparse
import json
import math
import sys
import tempfile
from functools import partial
from pathlib import Path

import datasets
import huggingface_hub as hf
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator
from models import load_checkpoint_state
from utils.import_utils import import_model
from utils.torch_utils import fixed_linear


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_statistics(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.float()
    mask = mask.to(device=x.device, dtype=x.dtype)[..., None]
    count = mask.sum().clamp_min(1.0)
    mean = (x * mask).sum(dim=(0, 1)) / count
    centered = x - mean
    covariance = torch.einsum(
        "bsi,bsj->ij", centered * mask, centered
    ) / count
    global_std = torch.sqrt(
        (centered.square() * mask).sum() / (count * x.shape[-1])
    )
    return mean, covariance, global_std


def masked_cross_correlation(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x.float()
    mask = mask.to(device=x.device, dtype=x.dtype)[..., None]
    count = mask.sum().clamp_min(1.0)
    return torch.einsum("bsi,bsj->ij", x * mask, x) / count


def cut_inv_sqrt(x: torch.Tensor, quantile: float) -> torch.Tensor:
    with torch.autocast(str(x.device.type), enabled=False):
        u, singular_values, vh = torch.linalg.svd(x.float())
        cutoff_index = round(quantile * (singular_values.shape[-1] - 1))
        cutoff = torch.sort(singular_values).values[cutoff_index]
        return u @ (
            torch.rsqrt(torch.maximum(singular_values, cutoff))[:, None] * vh
        )


def random_orthogonal(size: int, device: torch.device) -> torch.Tensor:
    q, r = torch.linalg.qr(
        torch.randn(size, size, device=device, dtype=torch.float32)
    )
    signs = torch.where(
        torch.diagonal(r) < 0,
        -torch.ones(size, device=device),
        torch.ones(size, device=device),
    )
    return q * signs[None]


@torch.no_grad()
def initialize_latent_input(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
    inv_quantile: float,
) -> None:
    """Use Piano's whitening/centering convention for each latent reader."""
    x = inputs[0].float()
    mean, covariance, _ = masked_statistics(x, mask)
    centered_whitening = cut_inv_sqrt(covariance, inv_quantile)
    cross_whitening = cut_inv_sqrt(
        masked_cross_correlation(x, mask), inv_quantile
    )

    module.up_latent.weight.copy_(
        (
            random_orthogonal(x.shape[-1], x.device) @ cross_whitening
        )[:module.latent_size].to(module.up_latent.weight.dtype)
    )
    for projection in (module.gate_latent, module.sig_latent):
        projection.weight.copy_(
            (
                random_orthogonal(x.shape[-1], x.device)
                @ centered_whitening
            )[:module.latent_size].to(projection.weight.dtype)
        )
        projection.bias.copy_(
            -fixed_linear(mean, projection.weight).to(projection.bias.dtype)
        )


@torch.no_grad()
def initialize_latent_output(
    module: torch.nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Match Piano's random output scale to the decoder layer's base RMS."""
    _, _, global_std = masked_statistics(output, mask)
    reader = module.latent_mlp
    reader.down_latent.weight.copy_(
        torch.randn_like(reader.down_latent.weight)
        * (global_std / math.sqrt(reader.latent_size))
    )


@torch.no_grad()
def initialize_encoder_state(
    model: torch.nn.Module,
    _module: torch.nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Whiten the normalized causal states entering the bidirectional stack."""
    x = output.float()
    float_mask = mask.to(device=x.device, dtype=x.dtype)[..., None]
    count = float_mask.sum().clamp_min(1.0)
    mean = (x * float_mask).sum(dim=(0, 1)) / count
    variance = (
        (x - mean).square() * float_mask
    ).sum(dim=(0, 1)) / count
    inverse_std = torch.rsqrt(
        variance.clamp_min(model.config.rms_norm_eps)
    )
    model.encoder_state_shift.copy_(
        -mean.to(model.encoder_state_shift.dtype)
    )
    model.encoder_state_scale.copy_(
        inverse_std.to(model.encoder_state_scale.dtype) - 1.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--repo", default="aklein4/vae-init")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--inv-quantile", type=float, default=0.25)
    parser.add_argument(
        "--checkpoint", default="aklein4/Llama-3.2-1B-TPU"
    )
    parser.add_argument("--checkpoint-step", type=int, default=0)
    args, overrides = parser.parse_known_args()

    with initialize_config_dir(version_base=None, config_dir=str(SRC / "configs")):
        config = compose(config_name="default", overrides=overrides)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    attention_kernel = config.model.attention_kernel
    config.model.attention_kernel = None
    model = import_model(config.model.type)(config.model)
    load_checkpoint_state(
        model,
        args.checkpoint,
        args.checkpoint_step,
        strict=False,
        verbose=True,
    )
    model = model.float().to(DEVICE).eval()

    config.data.collator.kwargs["cluster_length"] = 1
    collator = HorizonCollator(**config.data.collator.kwargs)
    dataset = datasets.load_dataset(
        config.data.dataset.url, **config.data.dataset.kwargs
    )
    batch = collator(list(dataset.take(args.episodes)))
    input_ids = batch["input_ids"][:, 0].to(DEVICE)
    valid_mask = batch["attention_mask"][:, 0].to(DEVICE)

    handles = []
    for module in model.decoder_mlps():
        handles.append(
            module.latent_mlp.register_forward_pre_hook(
                partial(
                    initialize_latent_input,
                    mask=valid_mask,
                    inv_quantile=args.inv_quantile,
                )
            )
        )
        handles.append(
            module.register_forward_hook(
                partial(initialize_latent_output, mask=valid_mask)
            )
        )
    handles.append(
        model.encoder_bidirectional_norm.register_forward_hook(
            partial(initialize_encoder_state, model, mask=valid_mask)
        )
    )
    try:
        with torch.no_grad():
            zero_noise = torch.zeros(
                input_ids.shape[0],
                config.model.latent_size,
                config.model.latent_size,
                device=DEVICE,
            )
            model(
                input_ids,
                valid_mask,
                alpha=torch.ones((), device=DEVICE),
                noise=zero_noise,
            )
    finally:
        for handle in handles:
            handle.remove()

    config.model.attention_kernel = attention_kernel
    state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }

    def save_checkpoint(save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "config.json", "w") as file:
            json.dump(
                OmegaConf.to_container(config.model, resolve=True),
                file,
                indent=4,
            )
        torch.save(state, save_dir / "model.pt")

    if args.output is not None:
        save_checkpoint(args.output / f"{args.step:012d}")

    if args.repo:
        hf.create_repo(args.repo, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary:
            save_dir = Path(temporary)
            save_checkpoint(save_dir)
            hf.HfApi().upload_folder(
                repo_id=args.repo,
                repo_type="model",
                folder_path=save_dir,
                path_in_repo=f"{args.step:012d}",
            )

    print(
        f"Initialized VAE from {args.episodes} episodes at step {args.step}"
    )


if __name__ == "__main__":
    main()
