from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import load_checkpoint
from data.datasets import get_dataset
from utils.import_utils import import_collator
from models.recurrent import RecurrentModel, RecurrentMode, DynamicLR
import utils.constants as constants


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DATA_CONFIG = "data/horizons-llama3.yaml"

SAVE_DIR = constants.LOCAL_DATA_PATH / "dynamic_lr"

FPS = 4
DPI = 75
FIGSIZE = (6.2, 5.4)
CMAP = "bwr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a GIF of one FastWeightMLP get_lr offset matrix over a horizon trajectory."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--data-config", type=str, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--autocast-dtype", default="bfloat16")
    return parser.parse_args()


def load_model(
    checkpoint: str,
    step: int,
) -> RecurrentModel:

    model: RecurrentModel = load_checkpoint(
        checkpoint, step,
        attention_kernel=("gpu_flash_attention" if DEVICE.type == "cuda" else None),
    )  

    model.to(device=DEVICE, dtype=torch.float32)
    model.train()

    for param in model.parameters():
        param.requires_grad_(False)

    model.model.embed_tokens.requires_grad_(True)
    model.gradient_checkpointing_enable()

    return model


def loss_fn(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids[:, 1:].contiguous()
    mask = assistant_mask[:, 1:].float().contiguous()

    losses = F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        labels.view(-1),
        reduction="none",
    ).view_as(labels)

    output_losses = (losses * mask).sum(1) / mask.sum(1).clamp(min=1)
    
    return output_losses.mean()


def get_layer_lr_module(model: RecurrentModel, layer: int) -> DynamicLR:
    return model._layer_module(layer, "mlp.dynamic_lr")


@torch.no_grad()
def get_log_lr(dynamic_lr: DynamicLR, embeds: torch.Tensor, embed_mask: torch.Tensor) -> torch.Tensor:
    return dynamic_lr.forward(embeds, embed_mask).log10()


def load_trajectory(data_config: str, index: int) -> dict[str, torch.Tensor]:

    config_path = constants.CONFIG_PATH(data_config)
    config = OmegaConf.load(config_path)

    dataset = get_dataset(config.dataset.url, config.dataset.kwargs)

    collator = import_collator(config.collator.type)(
        **config.collator.kwargs
    )

    i = 0
    batch = None
    for row in dataset:
        if i == index:
            batch = row
            break
        i += 1

    batch = collator([batch])
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    return config.dataset.url, batch


def collect_offsets(
    model: RecurrentModel,
    dynamic_lr: DynamicLR,
    batch: dict[str, torch.Tensor],
    autocast_dtype: torch.dtype,
) -> np.ndarray:
    input_ids = batch["input_ids"]
    assistant_mask = batch["assistant_mask"]
    attention_mask = batch["attention_mask"]

    model.init_state(input_ids.shape[0], DEVICE)

    offsets = []
    for horizon_idx in tqdm(range(input_ids.shape[1]), desc="Collecting offsets"):
        example_ids = input_ids[:, horizon_idx, :]
        example_mask = assistant_mask[:, horizon_idx, :]
        example_attn = attention_mask[:, horizon_idx, :]

        with torch.enable_grad(), torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):

            hidden_states = model.forward_backbone(
                input_ids=example_ids, mode=RecurrentMode.TRAIN_FIRST
            )

            lm_states = model.model.norm(hidden_states[:, :-1])
            loss = loss_fn(
                model.lm_head(lm_states),
                example_ids,
                example_mask,
            )

            with torch.no_grad():
                embeds = model.forward_embeddings(
                    hidden_states.detach(), example_attn
                )

        loss.backward()
        
        with torch.no_grad():

            log_lr = get_log_lr(dynamic_lr, embeds, example_attn)[0].float()
            offsets.append(log_lr.cpu().numpy())

            with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
                model.update_state(
                    embeds, example_mask, RecurrentMode.TRAIN_FIRST
                )

    model.empty_state()
    model.zero_grad(set_to_none=True)

    return np.stack(offsets)


def center_and_scale_offsets(offsets: np.ndarray) -> np.ndarray:
    return offsets - offsets.mean(axis=0, keepdims=True)


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

    plt.savefig(output, dpi=DPI)


def color_limits(offsets: np.ndarray) -> tuple[float, float]:
    mx = np.max(np.abs(offsets))
    return -mx, mx


def save_gif(offsets: np.ndarray, output: Path, metadata: dict) -> None:
    vmin, vmax = color_limits(offsets)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(offsets[0], cmap=CMAP, norm=norm, interpolation="nearest")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("log10 lr (centered)")
    title = ax.set_title("")
    ax.set_xlabel("input dimension")
    ax.set_ylabel("output dimension")

    def update(frame_idx: int):
        im.set_data(offsets[frame_idx])
        title.set_text(
            f"Layer {metadata['layer']} learning rates, episode {(frame_idx + 1):02d}/{len(offsets)}"
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
    autocast_dtype = getattr(torch, args.autocast_dtype)

    model = load_model(args.checkpoint, args.checkpoint_step)
    dynamic_lr = get_layer_lr_module(model, args.layer)
    data_url, batch = load_trajectory(args.data_config, args.index)

    offsets = collect_offsets(model, dynamic_lr, batch, autocast_dtype)

    ch = args.checkpoint.replace("/", "--")
    folder = (
        SAVE_DIR / 
        f"{ch}_step={args.checkpoint_step}_layer={args.layer}_trajectory={args.index}"
    )
    os.makedirs(folder, exist_ok=True)

    plot_average_offsets(offsets, folder / "average_offsets.png")

    offsets = center_and_scale_offsets(offsets)
    print_explained_variance(offsets)
    
    metadata = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "dataset": data_url,
        "trajectory_index": args.index,
        "layer": args.layer,
        "frames": int(offsets.shape[0]),
        "matrix_shape": list(offsets.shape[1:]),
        "autocast_dtype": args.autocast_dtype,
        "centered_by": "per_element_trajectory_mean",
        "scale": "log10",
    }

    save_gif(offsets, folder / "animation.gif", metadata)
    write_metadata(folder / "metadata.json", metadata)


if __name__ == "__main__":
    main()
