from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from collators.horizon import HorizonCollator
from models.ittt.configuration_ittt import ItttConfig
from models.ittt.modelling_ittt import FastWeightMLP, ItttModel
import utils.constants as constants


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "ittt_smollm2_360m_horizons.yaml"
DEFAULT_CHECKPOINT = "aklein4/iTTT-Cluster_horizons-recurrent-elu"
DEFAULT_CHECKPOINT_STEP = 800
DEFAULT_LAYER = 16
DEFAULT_TRAJECTORY_INDEX = 4
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if torch.cuda.is_available() else "eager"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "local_data"
    / f"lr_offset_layer_{DEFAULT_LAYER}_step_{DEFAULT_CHECKPOINT_STEP}.gif"
)

FPS = 4
DPI = 110
FIGSIZE = (6.2, 5.4)
CMAP = "bwr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a GIF of one FastWeightMLP get_lr offset matrix over a horizon trajectory."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-step", type=int, default=DEFAULT_CHECKPOINT_STEP)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--trajectory-index", type=int, default=DEFAULT_TRAJECTORY_INDEX)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--autocast-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--aux-weight", type=float, default=0.0, help="Weight for the auxiliary loss in the cluster loss computation.")
    return parser.parse_args()


def step_name(step: int) -> str:
    return f"step_{step:012d}"


def local_checkpoint(checkpoint: str, step: int) -> Path:
    return Path(constants.LOCAL_DATA_PATH) / Path(checkpoint).name / step_name(step)


def load_model(
    checkpoint: str,
    step: int,
    attn_implementation: str | None,
    device: torch.device,
) -> ItttModel:
    local_path = local_checkpoint(checkpoint, step)
    subfolder = step_name(step)

    if local_path.exists():
        print(f"Loading checkpoint from {local_path}")
        config = ItttConfig.from_pretrained(str(local_path))
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        model = ItttModel.from_pretrained(str(local_path), config=config)
    else:
        print(f"Loading checkpoint from {checkpoint}/{subfolder}")
        config = ItttConfig.from_pretrained(checkpoint, subfolder=subfolder)
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        model = ItttModel.from_pretrained(checkpoint, subfolder=subfolder, config=config)

    model.to(device=device, dtype=torch.float32)
    model.train()
    model.config.use_cache = False
    model.llama.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def cluster_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    aux_weight: float,
) -> torch.Tensor:
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = assistant_mask[:, 1:].float().contiguous()

    shift_attn = attention_mask[:, 1:].float().contiguous()
    shift_aux = (1.0 - shift_mask) * shift_attn

    losses = F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    output_losses = (losses * shift_mask).sum(1) / shift_mask.sum(1).clamp(min=1)
    aux_losses = (losses * shift_aux).sum(1) / shift_aux.sum(1).clamp(min=1)
    
    return output_losses.mean() + aux_weight * aux_losses.mean()


def get_layer_mlp(model: ItttModel, layer: int) -> FastWeightMLP:
    layers = model.llama.model.layers
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"layer must be in [0, {len(layers) - 1}], got {layer}")

    mlp = layers[layer].mlp
    if not isinstance(mlp, FastWeightMLP):
        raise TypeError(f"layer {layer} mlp is {type(mlp).__name__}, expected FastWeightMLP")
    return mlp


@torch.no_grad()
def get_lr_offset(mlp: FastWeightMLP, embeds: torch.Tensor, embed_mask: torch.Tensor) -> torch.Tensor:
    embeds = embeds * embed_mask[..., None].to(embeds.dtype)
    count = embed_mask.to(embeds.dtype).sum(-1).clamp_min(1.0)

    # [B, F, F]
    offset = (
        mlp.fast_p_l(embeds).transpose(-2, -1) @ mlp.fast_p_r(embeds) /
        count[..., None, None]
    )
    return -F.elu(-offset)


def load_trajectory(cfg, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    collator_kwargs = OmegaConf.to_container(cfg.collator.kwargs, resolve=True)
    collator = HorizonCollator(**collator_kwargs)

    split = cfg.dataset.kwargs.get("split", "train")
    dataset = datasets.load_dataset(cfg.dataset.name, split=split, streaming=True)
    rows = list(islice(dataset, args.trajectory_index, args.trajectory_index + 1))
    if not rows:
        raise RuntimeError(
            f"dataset {cfg.dataset.name}/{split} did not contain trajectory {args.trajectory_index}"
        )
    return collator(rows)


def collect_offsets(
    model: ItttModel,
    layer_mlp: FastWeightMLP,
    batch: dict[str, torch.Tensor],
    max_frames: int | None,
    autocast_dtype: torch.dtype,
    device: torch.device,
    aux_weight,
) -> np.ndarray:
    input_ids = batch["input_ids"]
    assistant_mask = batch["assistant_mask"]
    attention_mask = batch["attention_mask"]

    if input_ids.shape[0] != 1:
        raise ValueError(f"expected a single trajectory batch, got batch size {input_ids.shape[0]}")
    horizon_length = input_ids.shape[1]
    if max_frames is not None:
        horizon_length = min(horizon_length, max_frames)

    model.init_state(input_ids.shape[0], device)
    model.empty_state()
    model.disable_second_pass()
    model.zero_grad(set_to_none=True)

    offsets = []
    for horizon_idx in tqdm(range(horizon_length), desc="Collecting offsets"):
        example_ids = input_ids[:, horizon_idx, :]
        example_mask = assistant_mask[:, horizon_idx, :]
        example_attn = attention_mask[:, horizon_idx, :]

        with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
            outputs = model(example_ids, logits_to_keep=slice(0, -1))
            loss = cluster_loss(outputs.logits, example_ids, example_mask, example_attn, aux_weight)

            with torch.no_grad():
                embeds = model.bidirectional_forward(outputs.hidden_states, example_attn)
                offset = get_lr_offset(layer_mlp, embeds, example_attn)
                offsets.append(offset[0].float().cpu().numpy())

        loss.backward()
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            model.update_state(embeds, example_attn)
        model.zero_grad(set_to_none=True)

    model.empty_state()
    return np.stack(offsets)


def center_and_scale_offsets(offsets: np.ndarray) -> np.ndarray:
    return (offsets - offsets.mean(axis=0, keepdims=True)) / np.log(10.0)


def print_explained_variance(offsets: np.ndarray, n_components: int = 10) -> None:
    samples = offsets.reshape(offsets.shape[0], -1)
    samples = samples - samples.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(samples, full_matrices=False)
    explained_variance = (singular_values ** 2) / max(samples.shape[0] - 1, 1)
    explained_variance /= explained_variance.sum()

    values = explained_variance[:n_components]

    print(f"Explained variance of first {len(values)} principal components:")
    for i, value in enumerate(values):
        print(f"  PC {i + 1}: {value:.6f}")


def plot_average_offsets(offsets: np.ndarray, output: Path) -> None:

    avg = offsets.mean(axis=(-2, -1))

    plt.plot(avg, marker=".", markersize=10)
    plt.title("Average get_lr offset over horizon trajectory")
    plt.xlabel("Horizon step")
    plt.ylabel("Average get_lr offset")
    plt.grid()

    plt.savefig(output.with_name(output.stem + "_averages.png"), dpi=DPI)


def color_limits(offsets: np.ndarray) -> tuple[float, float]:
    mn = float(np.min(offsets))
    mx = float(np.max(offsets))
    return mn, mx


def save_gif(offsets: np.ndarray, output: Path, metadata: dict) -> None:
    vmin, vmax = color_limits(offsets)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(offsets[0], cmap=CMAP, norm=norm, interpolation="nearest")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("centered get_lr offset / ln(10)")
    title = ax.set_title("")
    ax.set_xlabel("fast_p_r dimension")
    ax.set_ylabel("fast_p_l dimension")

    def update(frame_idx: int):
        im.set_data(offsets[frame_idx])
        title.set_text(
            f"Layer {metadata['layer']} offset, episode {frame_idx + 1}/{len(offsets)}"
        )
        return im, title

    update(0)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(offsets), interval=1000 / FPS, blit=False)
    animation.save(output, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"Wrote {output}")


def write_metadata(output: Path, metadata: dict) -> None:
    path = output.with_suffix(".json")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    autocast_name = args.autocast_dtype or cfg.trainer.autocast_dtype
    autocast_dtype = getattr(torch, autocast_name)

    model = load_model(args.checkpoint, args.checkpoint_step, args.attn_implementation, device)
    layer_mlp = get_layer_mlp(model, args.layer)
    batch = load_trajectory(cfg, args)
    offsets = collect_offsets(model, layer_mlp, batch, args.max_frames, autocast_dtype, device, args.aux_weight)
    plot_average_offsets(offsets, args.output)
    offsets = center_and_scale_offsets(offsets)
    print_explained_variance(offsets)
    
    metadata = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "config": str(args.config),
        "dataset": cfg.dataset.name,
        "split": cfg.dataset.kwargs.get("split", "train"),
        "trajectory_index": args.trajectory_index,
        "layer": args.layer,
        "frames": int(offsets.shape[0]),
        "matrix_shape": list(offsets.shape[1:]),
        "autocast_dtype": autocast_name,
        "centered_by": "per_element_trajectory_mean",
        "scale": "1 / ln(10)",
        "aux_weight": args.aux_weight,
    }

    save_gif(offsets, args.output, metadata)
    write_metadata(args.output, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
