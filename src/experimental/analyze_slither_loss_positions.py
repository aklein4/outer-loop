from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn.functional as F

from data.datasets import get_dataset
from models import load_checkpoint
from utils import constants
from utils.import_utils import import_collator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Slither next-token loss as a function of position."
    )
    parser.add_argument("--checkpoint", default="aklein4/slither_alpha-350m")
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument("--data-config", default="data/longattn-smollm2.yaml")
    parser.add_argument("--trainer-config", default="trainer/slither-med.yaml")
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--boundary-radius", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=constants.LOCAL_DATA_PATH / "slither_loss_positions",
    )
    return parser.parse_args()


def load_tokens(data_config, num_examples: int):
    dataset = get_dataset(data_config.dataset.url, data_config.dataset.kwargs)
    iterator = iter(dataset)
    rows = [next(iterator) for _ in range(num_examples)]
    collator = import_collator(data_config.collator.type)(
        **data_config.collator.kwargs
    )
    tokens = collator(rows)["input_ids"]
    del iterator, dataset, rows
    gc.collect()
    return tokens


@torch.inference_mode()
def capture_losses(model, tokens, batch_size: int, use_autocast: bool):
    all_losses = []
    chunk_length = int(model.chunk_length)
    started = time.time()

    for batch_start in range(0, len(tokens), batch_size):
        batch = tokens[batch_start : batch_start + batch_size].to("cuda")
        input_chunks = list(batch[:, :-1].split(chunk_length, dim=1))
        label_chunks = list(batch[:, 1:].split(chunk_length, dim=1))
        model.init_state(batch.shape[0], batch.device)
        previous_mem = None
        batch_losses = []

        for chunk_idx, (input_ids, labels) in enumerate(
            zip(input_chunks, label_chunks)
        ):
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                logits, new_mem = model(
                    input_ids=input_ids,
                    mem_states=previous_mem,
                )
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
            batch_losses.append(losses.float().cpu().numpy())

            new_mem = new_mem.float()
            if chunk_idx < len(input_chunks) - 1:
                model.increment_state(new_mem)
            previous_mem = new_mem

        model.empty_state()
        all_losses.append(np.concatenate(batch_losses, axis=1))
        torch.cuda.synchronize()
        print(
            f"examples {batch_start:03d}–{batch_start + len(batch) - 1:03d}, "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    return np.concatenate(all_losses, axis=0)


def descriptive(values: np.ndarray) -> dict[str, float | int]:
    x = values.astype(np.float64, copy=False).reshape(-1)
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "sem": float(x.std(ddof=1) / np.sqrt(x.size)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
    }


def make_absolute_stats(losses: np.ndarray):
    return pd.DataFrame(
        {
            "prediction_position": np.arange(losses.shape[1]),
            "mean_loss": losses.mean(0),
            "std_loss": losses.std(0),
            "sem_loss": losses.std(0, ddof=1) / np.sqrt(losses.shape[0]),
        }
    )


def make_chunk_stats(losses: np.ndarray, chunk_length: int):
    rows = []
    for chunk, start in enumerate(range(0, losses.shape[1], chunk_length)):
        stop = min(start + chunk_length, losses.shape[1])
        rows.append(
            {
                "chunk": chunk,
                "prediction_start": start,
                "prediction_stop_exclusive": stop,
                **descriptive(losses[:, start:stop]),
            }
        )
    return pd.DataFrame(rows)


def make_within_chunk_stats(losses: np.ndarray, chunk_length: int):
    rows = []
    num_chunks = int(np.ceil(losses.shape[1] / chunk_length))
    for scope, chunk_indices in (
        ("all_chunks", range(num_chunks)),
        ("first_chunk", range(1)),
        ("later_chunks", range(1, num_chunks)),
    ):
        for position in range(chunk_length):
            arrays = []
            for chunk in chunk_indices:
                absolute = chunk * chunk_length + position
                if absolute < losses.shape[1]:
                    arrays.append(losses[:, absolute])
            if not arrays:
                continue
            rows.append(
                {
                    "scope": scope,
                    "within_chunk_position": position,
                    **descriptive(np.concatenate(arrays)),
                }
            )
    return pd.DataFrame(rows)


def make_boundary_profile(
    losses: np.ndarray, chunk_length: int, radius: int
):
    boundaries = np.arange(chunk_length, losses.shape[1], chunk_length)
    rows = []
    for offset in range(-radius, radius):
        values = np.stack([losses[:, boundary + offset] for boundary in boundaries])
        rows.append(
            {
                "offset": offset,
                "side": "before" if offset < 0 else "after",
                "example_clustered_sem": float(
                    values.mean(0).std(ddof=1) / np.sqrt(values.shape[1])
                ),
                **descriptive(values),
            }
        )
    return pd.DataFrame(rows), boundaries


def make_boundary_window_stats(
    losses: np.ndarray, boundaries: np.ndarray
):
    rows = []
    for width in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        before = np.stack(
            [losses[:, boundary - width : boundary].mean(1) for boundary in boundaries]
        )
        after = np.stack(
            [losses[:, boundary : boundary + width].mean(1) for boundary in boundaries]
        )
        difference = after - before
        per_example_difference = difference.mean(0)
        aggregate_relative_delta = float(
            difference.mean() / before.mean()
        )
        rows.append(
            {
                "window_tokens": width,
                "before_mean_loss": float(before.mean()),
                "after_mean_loss": float(after.mean()),
                "absolute_delta": float(difference.mean()),
                "delta_sem": float(
                    per_example_difference.std(ddof=1)
                    / np.sqrt(per_example_difference.size)
                ),
                "boundary_pair_delta_sem": float(
                    difference.std(ddof=1) / np.sqrt(difference.size)
                ),
                "relative_delta": aggregate_relative_delta,
                "fraction_after_gt_before": float(np.mean(difference > 0)),
                "boundary_example_pairs": int(difference.size),
            }
        )
    return pd.DataFrame(rows)


def make_boundary_by_chunk_stats(
    losses: np.ndarray, boundaries: np.ndarray
):
    rows = []
    for new_chunk, boundary in enumerate(boundaries, start=1):
        before = losses[:, boundary - 1]
        after = losses[:, boundary]
        difference = after - before
        rows.append(
            {
                "new_chunk": new_chunk,
                "boundary_prediction_position": int(boundary),
                "before_mean_loss": float(before.mean()),
                "after_mean_loss": float(after.mean()),
                "absolute_delta": float(difference.mean()),
                "delta_sem": float(
                    difference.std(ddof=1) / np.sqrt(difference.size)
                ),
                "fraction_after_gt_before": float(np.mean(difference > 0)),
            }
        )
    return pd.DataFrame(rows)


def make_position_windows(
    losses: np.ndarray, chunk_length: int
):
    num_chunks = int(np.ceil(losses.shape[1] / chunk_length))
    later_chunks = range(1, num_chunks)
    windows = (
        (0, 1),
        (1, 2),
        (2, 4),
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 768),
        (768, 960),
        (960, 1024),
    )
    rows = []
    for start, stop in windows:
        arrays = []
        for chunk in later_chunks:
            absolute_start = chunk * chunk_length + start
            absolute_stop = min(chunk * chunk_length + stop, losses.shape[1])
            if absolute_start < absolute_stop:
                arrays.append(losses[:, absolute_start:absolute_stop].reshape(-1))
        rows.append(
            {
                "within_chunk_start": start,
                "within_chunk_stop_exclusive": stop,
                **descriptive(np.concatenate(arrays)),
            }
        )
    return pd.DataFrame(rows)


def analyze_boundary_target_composition(
    tokens: torch.Tensor,
    losses: np.ndarray,
    boundaries: np.ndarray,
    chunk_length: int,
    vocab_size: int,
):
    targets = tokens[:, 1:].numpy()
    positions = np.arange(losses.shape[1])
    within = positions % chunk_length
    interior_mask = (
        (positions >= chunk_length)
        & (within >= 256)
        & (within < 960)
    )
    interior_targets = targets[:, interior_mask].reshape(-1)
    interior_losses = losses[:, interior_mask].reshape(-1)
    counts = np.bincount(interior_targets, minlength=vocab_size)
    sums = np.bincount(
        interior_targets, weights=interior_losses, minlength=vocab_size
    )
    token_mean_loss = np.divide(
        sums,
        counts,
        out=np.full(vocab_size, np.nan),
        where=counts > 0,
    )

    result = {}
    for name, selected_positions in (
        ("before_boundary", boundaries - 1),
        ("after_boundary", boundaries),
    ):
        selected_targets = targets[:, selected_positions].reshape(-1)
        selected_losses = losses[:, selected_positions].reshape(-1)
        expected = token_mean_loss[selected_targets]
        covered = np.isfinite(expected)
        result[name] = {
            "count": int(selected_targets.size),
            "actual_mean_loss": float(selected_losses.mean()),
            "token_composition_expected_interior_loss": float(
                expected[covered].mean()
            ),
            "interior_token_mean_coverage": float(covered.mean()),
            "eos_token_fraction": float(np.mean(selected_targets == 0)),
            "pad_token_fraction": float(
                np.mean(selected_targets == vocab_size - 1)
            ),
            "unique_target_tokens": int(np.unique(selected_targets).size),
        }
    result["interior_reference"] = {
        "count": int(interior_losses.size),
        "actual_mean_loss": float(interior_losses.mean()),
        "unique_target_tokens": int(np.unique(interior_targets).size),
    }
    return result


def plot_boundary_profile(profile: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    smooth = profile["mean"].rolling(9, center=True, min_periods=1).mean()
    ci = 1.96 * profile["example_clustered_sem"]
    ax.plot(profile["offset"], profile["mean"], alpha=0.3, linewidth=0.8)
    ax.plot(profile["offset"], smooth, linewidth=2, label="9-token moving average")
    ax.fill_between(
        profile["offset"],
        profile["mean"] - ci,
        profile["mean"] + ci,
        alpha=0.15,
        label="95% CI, clustered by example",
    )
    ax.axvline(-0.5, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="prediction offset from boundary (0 = first in new chunk)",
        ylabel="mean next-token cross-entropy",
        title="Loss around 1,024-token chunk boundaries",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_within_chunk(within: pd.DataFrame, output: Path):
    frame = within[within.scope == "later_chunks"].copy()
    smooth = frame["mean"].rolling(17, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        frame["within_chunk_position"],
        frame["mean"],
        alpha=0.25,
        linewidth=0.7,
        label="per-position mean",
    )
    ax.plot(
        frame["within_chunk_position"],
        smooth,
        linewidth=2,
        label="17-token moving average",
    )
    ax.set(
        xlabel="position within chunk",
        ylabel="mean next-token cross-entropy",
        title="Loss by position within recurrent chunks 1–31",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_chunk_means(chunk_stats: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(chunk_stats["chunk"], chunk_stats["mean"], marker=".")
    ax.set(
        xlabel="chunk index",
        ylabel="mean next-token cross-entropy",
        title="Average loss by chunk",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_boundary_by_chunk(frame: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(
        frame["new_chunk"],
        frame["absolute_delta"],
        yerr=1.96 * frame["delta_sem"],
        marker=".",
        linewidth=1,
        capsize=2,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="new chunk index",
        ylabel="loss(position 0) − loss(previous position)",
        title="Immediate loss discontinuity at each chunk boundary",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_absolute_position(absolute: pd.DataFrame, output: Path):
    smooth = absolute["mean_loss"].rolling(
        257, center=True, min_periods=1
    ).mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        absolute["prediction_position"],
        absolute["mean_loss"],
        alpha=0.12,
        linewidth=0.5,
    )
    ax.plot(
        absolute["prediction_position"],
        smooth,
        linewidth=1.5,
        label="257-token moving average",
    )
    ax.set(
        xlabel="absolute prediction position",
        ylabel="mean next-token cross-entropy",
        title="Loss over the 32K-token sequence",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This analysis requires the requested CUDA GPU.")

    data_config = OmegaConf.load(constants.CONFIG_PATH(args.data_config))
    trainer_config = OmegaConf.load(constants.CONFIG_PATH(args.trainer_config))
    output = args.output_dir / (
        f"{args.checkpoint.replace('/', '--')}_step={args.checkpoint_step}"
        f"_n={args.num_examples}"
    )
    output.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(data_config, args.num_examples)
    model = load_checkpoint(
        args.checkpoint,
        args.checkpoint_step,
        attention_kernel="gpu_flash_attention",
    )
    model.to(device="cuda", dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    losses = capture_losses(
        model,
        tokens,
        args.batch_size,
        bool(trainer_config.use_autocast),
    )
    chunk_length = int(model.chunk_length)

    absolute = make_absolute_stats(losses)
    chunks = make_chunk_stats(losses, chunk_length)
    within = make_within_chunk_stats(losses, chunk_length)
    boundary, boundaries = make_boundary_profile(
        losses, chunk_length, args.boundary_radius
    )
    boundary_windows = make_boundary_window_stats(losses, boundaries)
    boundary_by_chunk = make_boundary_by_chunk_stats(losses, boundaries)
    position_windows = make_position_windows(losses, chunk_length)
    token_composition = analyze_boundary_target_composition(
        tokens,
        losses,
        boundaries,
        chunk_length,
        int(model.config.vocab_size),
    )

    absolute.to_csv(output / "absolute_position_stats.csv", index=False)
    chunks.to_csv(output / "chunk_stats.csv", index=False)
    within.to_csv(output / "within_chunk_position_stats.csv", index=False)
    boundary.to_csv(output / "boundary_profile.csv", index=False)
    boundary_windows.to_csv(output / "boundary_window_comparisons.csv", index=False)
    boundary_by_chunk.to_csv(output / "boundary_by_chunk.csv", index=False)
    position_windows.to_csv(output / "within_chunk_window_stats.csv", index=False)
    np.save(output / "per_example_token_losses.npy", losses)
    (output / "boundary_target_composition.json").write_text(
        json.dumps(token_composition, indent=2) + "\n"
    )

    plot_boundary_profile(boundary, output / "boundary_profile.png")
    plot_within_chunk(within, output / "within_chunk_profile.png")
    plot_chunk_means(chunks, output / "chunk_mean_losses.png")
    plot_boundary_by_chunk(boundary_by_chunk, output / "boundary_by_chunk.png")
    plot_absolute_position(absolute, output / "absolute_position_loss.png")

    later = within[within.scope == "later_chunks"].set_index(
        "within_chunk_position"
    )
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "dataset": data_config.dataset.url,
        "num_examples": args.num_examples,
        "sequence_length": int(data_config.collator.kwargs.sequence_length),
        "prediction_positions": int(losses.shape[1]),
        "chunk_length": chunk_length,
        "num_chunks": int(np.ceil(losses.shape[1] / chunk_length)),
        "num_boundaries": int(len(boundaries)),
        "parameter_dtype": "float32",
        "autocast": bool(trainer_config.use_autocast),
        "autocast_dtype": "bfloat16",
        "attention_kernel": "gpu_flash_attention",
        "device": torch.cuda.get_device_name(),
        "overall_loss": descriptive(losses),
        "later_chunk_selected_positions": {
            str(position): {
                "mean": float(later.loc[position, "mean"]),
                "sem": float(later.loc[position, "sem"]),
            }
            for position in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 960, 1023)
            if position in later.index
        },
        "boundary_window_comparisons": boundary_windows.to_dict(orient="records"),
        "boundary_target_composition": token_composition,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote analysis to {output}", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
