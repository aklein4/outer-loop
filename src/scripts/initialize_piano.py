"""Activation-initialize a Piano model and upload it as a checkpoint."""

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-channel and global statistics over unmasked tokens."""

    x = x.float()
    mask = mask.to(device=x.device, dtype=x.dtype)[..., None]
    count = mask.sum().clamp_min(1.0)

    mean = (x * mask).sum(dim=(0, 1)) / count
    centered = x - mean
    variance = (centered.square() * mask).sum(dim=(0, 1)) / count
    covariance = torch.einsum(
        "bsi,bsj->ij", centered * mask, centered
    ) / count
    global_std = torch.sqrt(variance.mean())

    return mean, torch.sqrt(variance), covariance, global_std


def cut_inv_sqrt(
    x: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Return a quantile-clipped inverse square root of a PSD matrix."""

    with torch.autocast(str(x.device.type), enabled=False):
        x = x.float()
        u, singular_values, vh = torch.linalg.svd(x)
        sorted_values = torch.sort(singular_values).values
        rank = round(quantile * (singular_values.shape[-1] - 1))
        cutoff = sorted_values[rank]
        singular_values = torch.maximum(singular_values, cutoff)
        return u @ (torch.rsqrt(singular_values)[:, None] * vh)


def cut_reciprocal(
    x: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Return quantile-clipped reciprocals of nonnegative values."""

    x = x.float()
    sorted_values = torch.sort(x).values
    rank = round(quantile * (x.shape[-1] - 1))
    cutoff = sorted_values[rank]
    return torch.reciprocal(torch.maximum(x, cutoff))


def random_orthogonal(size: int, device: torch.device) -> torch.Tensor:
    """Generate a Haar-distributed orthogonal matrix."""

    q, r = torch.linalg.qr(
        torch.randn(size, size, device=device, dtype=torch.float32)
    )
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), 1.0)
    return q * signs[None]


@torch.no_grad()
def initialize_fast_input(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
    inv_quantile: float,
) -> None:
    """Whiten a fast MLP's input projections before its calibration pass."""

    x = inputs[0].float()
    mean, _, covariance, _ = masked_statistics(x, mask)
    whitening = cut_inv_sqrt(covariance, inv_quantile)

    for projection in (module.up_fast, module.gate_fast, module.sig_fast):
        weight = (
            random_orthogonal(x.shape[-1], x.device) @ whitening
        )[:module.fast_weight_size]
        projection.weight.copy_(weight.to(projection.weight.dtype))
        projection.bias.copy_(
            -fixed_linear(mean, projection.weight).to(projection.bias.dtype)
        )


@torch.no_grad()
def initialize_fast_output(
    module: torch.nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Scale a fast MLP's output projection from its base output RMS."""

    _, _, _, global_std = masked_statistics(output, mask)
    module.down_fast.weight.copy_(
        torch.randn_like(module.down_fast.weight)
        * (global_std / math.sqrt(module.fast_weight_size))
    )


@torch.no_grad()
def scale_gate_input(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
    inv_quantile: float,
) -> None:
    """Scale a scalar gate by clipped inverse input-channel stds."""

    _, std, _, _ = masked_statistics(inputs[0], mask)
    inverse_std = cut_reciprocal(std, inv_quantile)
    module.weight.mul_(inverse_std.to(module.weight.dtype)[None])


@torch.no_grad()
def initialize_pooler_input(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
    inv_quantile: float,
) -> None:
    """Whiten and independently rotate a soft pooler's input projections."""

    x = inputs[0].float()
    _, _, covariance, _ = masked_statistics(x, mask)
    whitening = cut_inv_sqrt(covariance, inv_quantile)

    for projection in (module.w_proj, module.v_proj):
        weight = (
            random_orthogonal(x.shape[-1], x.device) @ whitening
        )[:projection.out_features]
        projection.weight.copy_(weight.to(projection.weight.dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--repo", default="aklein4/piano-init")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--inv-quantile", type=float, default=0.25)
    parser.add_argument("--checkpoint", type=str, default="aklein4/Llama-3.2-1B-TPU")
    parser.add_argument("--checkpoint-step", type=int, default=0)
    args, overrides = parser.parse_known_args()

    with initialize_config_dir(version_base=None, config_dir=str(SRC / "configs")):
        config = compose(config_name="default", overrides=overrides)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    attention_kernel = config.model.attention_kernel
    config.model.attention_kernel = None
    model = import_model(config.model.type)(config.model)
    checkpoint = args.checkpoint or config.model.pretrained_url
    checkpoint_step = (
        args.checkpoint_step
        if args.checkpoint is not None
        else config.model.pretrained_step
    )
    if checkpoint is not None:
        load_checkpoint_state(
            model,
            checkpoint,
            checkpoint_step,
            strict=(
                False
                if args.checkpoint is not None
                else config.model.pretrained_strict
            ),
        )
    model = model.float().to(DEVICE).eval()

    config.data.collator.kwargs["cluster_length"] = 1
    collator = HorizonCollator(**config.data.collator.kwargs)

    data = datasets.load_dataset(
        config.data.dataset.url,
        **config.data.dataset.kwargs,
    )
    rows = list(data.take(args.episodes))
    batch = collator(rows)

    input_ids = batch["input_ids"][:, 0].to(DEVICE)
    init_mask = batch["attention_mask"][:, 0].to(DEVICE)

    # The state is zero during calibration, so a singleton batch broadcasts to
    # every episode without allocating fast-weight state for every example.
    model.init_state(1, DEVICE)
    handles = []
    for module in model.fast_modules():
        handles.append(
            module.register_forward_pre_hook(
                partial(
                    initialize_fast_input,
                    mask=init_mask,
                    inv_quantile=args.inv_quantile,
                )
            )
        )
        handles.append(
            module.register_forward_hook(
                partial(initialize_fast_output, mask=init_mask)
            )
        )
        for projection in (
            module.fast_dynamic_lr.token_gate_proj,
            module.fast_dynamic_lr.next_token_gate_proj,
        ):
            handles.append(
                projection.register_forward_pre_hook(
                    partial(
                        scale_gate_input,
                        mask=init_mask,
                        inv_quantile=args.inv_quantile,
                    )
                )
            )
        handles.append(
            module.fast_dynamic_lr.sequence_token_gate.register_forward_pre_hook(
                partial(
                    initialize_pooler_input,
                    mask=init_mask,
                    inv_quantile=args.inv_quantile,
                )
            )
        )

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    config.model.attention_kernel = attention_kernel

    hf.create_repo(args.repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as save_dir:
        with open(Path(save_dir) / "config.json", "w") as f:
            json.dump(OmegaConf.to_container(config.model, resolve=True), f, indent=4)
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            Path(save_dir) / "model.pt",
        )
        hf.HfApi().upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=save_dir,
            path_in_repo=f"{args.step:012d}",
        )

    print(
        f"Uploaded {args.episodes} episode initialization "
        f"to {args.repo} step {args.step}"
    )


if __name__ == "__main__":
    main()
