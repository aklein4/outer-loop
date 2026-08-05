"""Probe a Forte ICL adaptation trajectory without running evaluation loops.

This is deliberately close to ``evaluate_icl.py``'s adaptation path.  For each
adaptation step it records per-trajectory, per-fast-layer magnitudes for the
state, the current raw ``G``, the gated update before the dynamic learning
rate, and the update actually applied to the state.  It also records cosines
to the state immediately before and immediately after the update.

The saved arrays are summaries (norms and cosines), not the full fast-weight
matrices.  Saving all matrices for a 1024-step trajectory would be needlessly
large and is not required for the requested magnitude plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import datasets


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_icl import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_TOKENIZER,
    adaptation_loss,
    autocast,
    encode,
    load_all_rows,
    load_model,
    load_tokenizer,
)
from models.forte import ForteMode, ForteModel  # noqa: E402
from utils import constants  # noqa: E402


DEFAULT_OUTPUT = constants.LOCAL_DATA_PATH / "forte_v3_step100_icl_state_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="aklein4/Horizon-TPU_forte-v3-1b"
    )
    parser.add_argument("--checkpoint-step", type=int, default=100)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=16,
        help="Number of trajectories sampled round-robin across subsets.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-adaptation-steps", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def norm_per_batch(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(1).norm(dim=1)


def cosine_per_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().flatten(1)
    y = y.float().flatten(1)
    x_norm = x.norm(dim=1)
    y_norm = y.norm(dim=1)
    denominator = x_norm * y_norm
    cosine = (x * y).sum(dim=1) / denominator.clamp_min(1e-30)
    return torch.where(denominator > 0, cosine, torch.full_like(cosine, float("nan")))


def select_rows(rows: list[dict], max_rows: int | None) -> list[dict]:
    if max_rows is None or max_rows <= 0 or max_rows >= len(rows):
        return rows

    by_subset: dict[str, list[dict]] = {}
    for row in rows:
        by_subset.setdefault(row["subset"], []).append(row)

    selected: list[dict] = []
    subsets = sorted(by_subset)
    while len(selected) < max_rows:
        added = False
        for subset in subsets:
            candidates = by_subset[subset]
            if candidates:
                selected.append(candidates.pop(0))
                added = True
                if len(selected) == max_rows:
                    break
        if not added:
            break
    return selected


def load_probe_rows(args: argparse.Namespace) -> list[dict]:
    """Load only the train trajectories needed by the adaptation probe."""
    subsets = args.subsets or datasets.get_dataset_config_names(args.dataset)
    loader_args = SimpleNamespace(
        dataset=args.dataset,
        num_examples=[args.num_adaptation_steps],
        num_eval=args.num_eval,
        max_rows=None,
    )
    rows = load_all_rows(loader_args, subsets)
    rows = select_rows(rows, args.max_rows)
    if not rows:
        raise RuntimeError("No eligible trajectories were found")
    return rows


def adapt_and_measure(
    model: ForteModel,
    tokenizer,
    batch_rows: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run one independent batch trajectory and return step x row x layer arrays."""
    if not isinstance(model, ForteModel):
        raise TypeError(f"This probe expects ForteModel, got {type(model).__name__}")

    num_steps = args.num_adaptation_steps
    num_rows = len(batch_rows)
    fast_modules = model.fast_modules()
    num_layers = len(fast_modules)
    metrics = {
        name: np.empty((num_steps, num_rows, num_layers), dtype=np.float32)
        for name in (
            "state_norm",
            "raw_G_norm",
            "pre_lr_update_norm",
            "applied_update_norm",
            "state_norm_after",
            "cos_pre_lr_update_vs_state",
            "cos_applied_update_vs_state",
            "cos_raw_G_vs_state",
            "cos_pre_lr_update_vs_raw_G",
            "cos_applied_update_vs_raw_G",
            "cos_applied_update_vs_state_after",
        )
    }
    losses = np.empty(num_steps, dtype=np.float32)

    model.init_state(num_rows, device)
    model.empty_state()
    try:
        for step in range(num_steps):
            input_ids, assistant_mask, attention_mask = encode(
                tokenizer,
                [row["train_data"][step] for row in batch_rows],
                args.max_length,
                device,
            )
            with torch.enable_grad():
                with torch.no_grad():
                    hidden_states = model.forward_backbone(
                        input_ids, mode=ForteMode.INFERENCE
                    )
                    embeddings = model.forward_embeddings(
                        hidden_states, attention_mask
                    )
                with autocast(device, args.dtype):
                    logits = model(
                        input_ids,
                        embeddings=embeddings,
                        embedding_mask=attention_mask,
                        mode=ForteMode.TRAIN_FIRST,
                        logits_to_keep=slice(0, -1),
                    )
                    loss = adaptation_loss(
                        input_ids,
                        assistant_mask,
                        attention_mask,
                        logits,
                        args.aux_weight,
                    )
                loss.backward()

            losses[step] = float(loss.detach().cpu())
            applied_updates: list[torch.Tensor] = []
            for layer, mlp in enumerate(fast_modules):
                state = mlp.state
                raw_g = mlp.grad_buffer.grad
                pre_lr_update = state.grad
                lr = mlp.fast_dynamic_lr(embeddings, attention_mask)
                applied_update = -lr * pre_lr_update * args.lr_scale
                applied_updates.append(applied_update)

                state_norm = norm_per_batch(state)
                raw_g_norm = norm_per_batch(raw_g)
                pre_lr_norm = norm_per_batch(pre_lr_update)
                applied_norm = norm_per_batch(applied_update)
                cos_pre_lr = cosine_per_batch(pre_lr_update, state)
                cos_applied = cosine_per_batch(applied_update, state)
                cos_raw = cosine_per_batch(raw_g, state)
                cos_pre_lr_raw = cosine_per_batch(pre_lr_update, raw_g)
                cos_applied_raw = cosine_per_batch(applied_update, raw_g)

                metrics["state_norm"][step, :, layer] = state_norm.detach().cpu().numpy()
                metrics["raw_G_norm"][step, :, layer] = raw_g_norm.detach().cpu().numpy()
                metrics["pre_lr_update_norm"][step, :, layer] = pre_lr_norm.detach().cpu().numpy()
                metrics["applied_update_norm"][step, :, layer] = applied_norm.detach().cpu().numpy()
                metrics["cos_pre_lr_update_vs_state"][step, :, layer] = cos_pre_lr.detach().cpu().numpy()
                metrics["cos_applied_update_vs_state"][step, :, layer] = cos_applied.detach().cpu().numpy()
                metrics["cos_raw_G_vs_state"][step, :, layer] = cos_raw.detach().cpu().numpy()
                metrics["cos_pre_lr_update_vs_raw_G"][step, :, layer] = cos_pre_lr_raw.detach().cpu().numpy()
                metrics["cos_applied_update_vs_raw_G"][step, :, layer] = cos_applied_raw.detach().cpu().numpy()

            model.update_state(
                embeddings,
                attention_mask,
                mode=ForteMode.TRAIN_FIRST,
                lr_scale=args.lr_scale,
            )
            for layer, (mlp, applied_update) in enumerate(zip(fast_modules, applied_updates)):
                state_after = mlp.state
                metrics["state_norm_after"][step, :, layer] = (
                    norm_per_batch(state_after).detach().cpu().numpy()
                )
                metrics["cos_applied_update_vs_state_after"][step, :, layer] = (
                    cosine_per_batch(applied_update, state_after).detach().cpu().numpy()
                )

            # The embedding table is the only explicitly trainable model
            # parameter in evaluate_icl.py. It is not optimized in this probe;
            # releasing its accumulated gradient keeps the long trajectory's
            # memory use bounded without changing the fast-state update.
            model.embed_tokens.weight.grad = None
            if step == 0 or (step + 1) % 32 == 0:
                print(
                    f"batch rows={num_rows} adaptation step {step + 1:04d}/{num_steps}: "
                    f"loss={losses[step]:.5f}",
                    flush=True,
                )
    finally:
        model.empty_state()
        model.embed_tokens.weight.grad = None

    return metrics, losses


def concatenate_batches(
    batch_metrics: list[dict[str, np.ndarray]],
    batch_losses: list[np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    names = batch_metrics[0].keys()
    metrics = {name: np.concatenate([x[name] for x in batch_metrics], axis=1) for name in names}
    losses = np.stack(batch_losses, axis=1)
    return metrics, losses


def write_npz(output_dir: Path, metrics: dict[str, np.ndarray], losses: np.ndarray) -> None:
    np.savez_compressed(output_dir / "metrics.npz", losses=losses, **metrics)


def write_csv(
    output_dir: Path,
    metrics: dict[str, np.ndarray],
    losses: np.ndarray,
    rows: list[dict],
) -> None:
    names = [
        "adaptation_step",
        "trajectory",
        "subset",
        "source",
        "layer",
        "batch_loss",
        *metrics.keys(),
    ]
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        num_steps, num_rows, num_layers = metrics["state_norm"].shape
        for step in range(num_steps):
            for trajectory, row in enumerate(rows):
                for layer in range(num_layers):
                    record = {
                        "adaptation_step": step + 1,
                        "trajectory": trajectory,
                        "subset": row["subset"],
                        "source": row.get("source", ""),
                        "layer": layer,
                        "batch_loss": float(losses[step].mean()),
                    }
                    for name, values in metrics.items():
                        record[name] = float(values[step, trajectory, layer])
                    writer.writerow(record)


def nan_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": math.nan, "median": math.nan, "q25": math.nan, "q75": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def nanmedian(values: np.ndarray, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def nanpercentile(values: np.ndarray, percentile: float, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(values, percentile, axis=axis)


def plot_magnitudes(output_dir: Path, metrics: dict[str, np.ndarray]) -> None:
    steps = np.arange(1, metrics["state_norm"].shape[0] + 1)
    specs = [
        ("state_norm", "State Frobenius norm", "State magnitude"),
        ("raw_G_norm", "Raw $G$ Frobenius norm", "Raw $G$ magnitude"),
        ("applied_update_norm", "Applied update Frobenius norm", "Applied update magnitude"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True)
    for ax, (key, ylabel, title) in zip(axes, specs):
        values = metrics[key]
        layer_median = nanmedian(values, axis=1)
        for layer in range(layer_median.shape[1]):
            y = layer_median[:, layer].copy()
            y[y <= 0] = np.nan
            ax.plot(steps, y, linewidth=0.9, alpha=0.45)
        flattened = values.reshape(values.shape[0], -1)
        overall = nanmedian(flattened, axis=1)
        q25 = nanpercentile(flattened, 25, axis=1)
        q75 = nanpercentile(flattened, 75, axis=1)
        overall[overall <= 0] = np.nan
        q25[q25 <= 0] = np.nan
        q75[q75 <= 0] = np.nan
        ax.plot(steps, overall, color="black", linewidth=2.0, label="all-layer median")
        ax.fill_between(steps, q25, q75, color="black", alpha=0.12, label="25–75%")
        ax.set_yscale("log")
        ax.set_xlabel("Adaptation step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.22, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Forte ICL adaptation magnitudes (colored lines are fast layers)")
    fig.tight_layout()
    fig.savefig(output_dir / "magnitudes_vs_adaptation_step.png", dpi=200)
    plt.close(fig)


def plot_cosines(output_dir: Path, metrics: dict[str, np.ndarray]) -> None:
    steps = np.arange(1, metrics["state_norm"].shape[0] + 1)
    specs = [
        ("cos_applied_update_vs_state", "cos(applied update, state before)"),
        ("cos_pre_lr_update_vs_state", "cos(pre-LR update, state before)"),
        ("cos_raw_G_vs_state", "cos(raw $G$, state before)"),
        ("cos_applied_update_vs_state_after", "cos(applied update, state after)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, specs):
        values = metrics[key]
        layer_median = nanmedian(values, axis=1)
        for layer in range(layer_median.shape[1]):
            ax.plot(steps, layer_median[:, layer], linewidth=0.9, alpha=0.45)
        flattened = values.reshape(values.shape[0], -1)
        overall = nanmedian(flattened, axis=1)
        q25 = nanpercentile(flattened, 25, axis=1)
        q75 = nanpercentile(flattened, 75, axis=1)
        ax.plot(steps, overall, color="black", linewidth=2.0, label="all-layer median")
        ax.fill_between(steps, q25, q75, color="black", alpha=0.12, label="25–75%")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Adaptation step")
        ax.set_ylabel("Cosine similarity")
        ax.set_title(title)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("Forte ICL adaptation update/state geometry")
    fig.tight_layout()
    fig.savefig(output_dir / "cosines_vs_adaptation_step.png", dpi=200)
    plt.close(fig)


def plot_heatmaps(output_dir: Path, metrics: dict[str, np.ndarray]) -> None:
    specs = [
        ("state_norm", "log10 median state norm"),
        ("raw_G_norm", "log10 median raw $G$ norm"),
        ("applied_update_norm", "log10 median applied update norm"),
        ("cos_applied_update_vs_state", "median cos(applied update, state before)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    for ax, (key, title) in zip(axes.flat, specs):
        values = metrics[key]
        layer_values = nanmedian(values, axis=1).T
        if key.endswith("norm"):
            layer_values = np.log10(np.maximum(layer_values, 1e-30))
        image = ax.imshow(
            layer_values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(1, values.shape[0], 0, values.shape[2]),
            cmap="viridis" if key.endswith("norm") else "coolwarm",
        )
        ax.set_title(title)
        ax.set_ylabel("Fast layer")
        ax.set_xlabel("Adaptation step")
        fig.colorbar(image, ax=ax, shrink=0.9)
    fig.suptitle("Layerwise adaptation trajectory summaries")
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_heatmaps.png", dpi=200)
    plt.close(fig)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    rows: list[dict],
    metrics: dict[str, np.ndarray],
    losses: np.ndarray,
    peak_memory_gb: float | None,
) -> None:
    num_steps, num_rows, num_layers = metrics["state_norm"].shape
    state_start_index = min(1, num_steps - 1)
    final_index = num_steps - 1
    final = {name: values[final_index] for name, values in metrics.items()}

    def global_stat(name: str, index: int = final_index) -> dict[str, float]:
        return nan_summary(metrics[name][index])

    def growth_stat(name: str) -> float:
        first = np.nanmedian(metrics[name][state_start_index])
        last = np.nanmedian(metrics[name][final_index])
        return float(last / max(first, 1e-30))

    cos_keys = (
        "cos_applied_update_vs_state",
        "cos_pre_lr_update_vs_state",
        "cos_raw_G_vs_state",
        "cos_pre_lr_update_vs_raw_G",
        "cos_applied_update_vs_raw_G",
        "cos_applied_update_vs_state_after",
    )
    cos_over_trajectory = {
        key: nan_summary(metrics[key][1:] if num_steps > 1 else metrics[key])
        for key in cos_keys
    }
    layer_final_state = np.nanmedian(final["state_norm"], axis=0)
    strongest_layers = np.argsort(layer_final_state)[::-1][:5].tolist()
    lines = [
        "# Forte v3 ICL state/update probe",
        "",
        "## Run",
        "",
        f"- Checkpoint: `{args.checkpoint}` at step `{args.checkpoint_step}`",
        f"- Device: `{args.device}`; dtype: `{args.dtype}`; torch-xla was not used",
        f"- Dataset: `{args.dataset}`",
        f"- Trajectories: `{num_rows}` sampled round-robin across `{len(set(row['subset'] for row in rows))}` subsets",
        f"- Batch size: `{args.batch_size}`; effective batches: `{math.ceil(num_rows / args.batch_size)}`",
        f"- Adaptation steps: `{num_steps}`; max sequence length: `{args.max_length}`",
        f"- Mean adaptation loss: `{float(losses.mean()):.6f}`; final-step loss: `{float(losses[-1].mean()):.6f}`",
    ]
    if peak_memory_gb is not None:
        lines.append(f"- Peak CUDA allocated memory: `{peak_memory_gb:.2f} GiB`")
    lines += [
        "",
        "## Metric definitions",
        "",
        "- `state_norm`: Frobenius norm of each fast-weight state immediately before the current adaptation update.",
        "- `raw_G_norm`: Frobenius norm of the current raw `G` in `grad_buffer.grad`, before it is accumulated into the gradient buffer.",
        "- `pre_lr_update_norm`: norm of `state.grad`, i.e. the normalized/gated update before applying the dynamic learning-rate matrix.",
        "- `applied_update_norm`: norm of the actual update `-lr * state.grad * lr_scale` added to the state.",
        "- Update/state cosines use the state immediately before the update. The first step is undefined because the state starts at zero and is stored as `NaN`; an after-update cosine is also included for reference.",
        "- All reported values are per trajectory and per fast layer; plots show the median across trajectories with colored layer traces.",
        "",
        "## Final-step magnitudes",
        "",
        "| quantity | mean | median | q25 | q75 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in (
        ("state_norm", "state norm"),
        ("raw_G_norm", "raw G norm"),
        ("pre_lr_update_norm", "pre-LR update norm"),
        ("applied_update_norm", "applied update norm"),
    ):
        summary = global_stat(name)
        lines.append(
            f"| {label} | {summary['mean']:.5g} | {summary['median']:.5g} | "
            f"{summary['q25']:.5g} | {summary['q75']:.5g} |"
        )
    lines += [
        "",
        "## Trajectory changes",
        "",
        f"Using step `{state_start_index + 1}` as the first nonzero-state comparison, the final/initial median ratios were: state `{growth_stat('state_norm'):.4g}`, raw G `{growth_stat('raw_G_norm'):.4g}`, pre-LR update `{growth_stat('pre_lr_update_norm'):.4g}`, and applied update `{growth_stat('applied_update_norm'):.4g}`.",
        f"The five layers with the largest final median state norms were: `{', '.join(str(layer) for layer in strongest_layers)}`.",
        "",
        "## Cosine summaries",
        "",
        "Cosines below exclude the undefined zero-state first step.",
        "",
        "| cosine | mean | median | q25 | q75 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("cos_applied_update_vs_state", "applied update vs state before"),
        ("cos_pre_lr_update_vs_state", "pre-LR update vs state before"),
        ("cos_raw_G_vs_state", "raw G vs state before"),
        ("cos_pre_lr_update_vs_raw_G", "pre-LR update vs raw G"),
        ("cos_applied_update_vs_raw_G", "applied update vs raw G"),
        ("cos_applied_update_vs_state_after", "applied update vs state after"),
    ):
        summary = cos_over_trajectory[key]
        lines.append(
            f"| {label} | {summary['mean']:.5g} | {summary['median']:.5g} | "
            f"{summary['q25']:.5g} | {summary['q75']:.5g} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `metrics.npz`: compressed step × trajectory × layer arrays.",
        "- `metrics.csv`: long-form copy of the same per-layer metrics.",
        "- `magnitudes_vs_adaptation_step.png`: state, raw G, and applied-update magnitudes.",
        "- `cosines_vs_adaptation_step.png`: update/state and raw-G/state cosine traces.",
        "- `layerwise_heatmaps.png`: layer-by-step summary heatmaps.",
        "- `metadata.json`: run arguments and selected trajectory metadata.",
        "",
        "No evaluation/test loops were run.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SRC, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.num_adaptation_steps < 1:
        raise ValueError("--num-adaptation-steps must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_probe_rows(args)
    print(f"Loaded {len(rows)} trajectories", flush=True)
    model = load_model(args.checkpoint, args.checkpoint_step, device)
    tokenizer = load_tokenizer(args.tokenizer)
    if args.compile:
        raise NotImplementedError("Compilation is intentionally not used by this probe")

    batch_metrics: list[dict[str, np.ndarray]] = []
    batch_losses: list[np.ndarray] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        print(
            f"Starting trajectory batch {start // args.batch_size + 1}/"
            f"{math.ceil(len(rows) / args.batch_size)} ({len(batch_rows)} rows)",
            flush=True,
        )
        current_metrics, current_losses = adapt_and_measure(
            model, tokenizer, batch_rows, args, device
        )
        batch_metrics.append(current_metrics)
        batch_losses.append(current_losses)

    metrics, losses = concatenate_batches(batch_metrics, batch_losses)
    write_npz(args.output_dir, metrics, losses)
    write_csv(args.output_dir, metrics, losses, rows)
    plot_magnitudes(args.output_dir, metrics)
    plot_cosines(args.output_dir, metrics)
    plot_heatmaps(args.output_dir, metrics)

    metadata = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "subsets": sorted(set(row["subset"] for row in rows)),
        "num_rows": len(rows),
        "batch_size": args.batch_size,
        "num_batches": len(batch_metrics),
        "num_adaptation_steps": args.num_adaptation_steps,
        "max_length": args.max_length,
        "device": str(device),
        "dtype": args.dtype,
        "lr_scale": args.lr_scale,
        "aux_weight": args.aux_weight,
        "torch_version": torch.__version__,
        "torch_xla_used": False,
        "git_revision": git_revision(),
        "rows": [
            {
                "trajectory": i,
                "subset": row["subset"],
                "source": row.get("source"),
                "num_train": len(row["train_data"]),
                "num_test": len(row["test_data"]),
            }
            for i, row in enumerate(rows)
        ],
        "loss_mean_by_step": losses.mean(axis=1).tolist(),
        "loss_mean": float(losses.mean()),
        "peak_cuda_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else None
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(
        args.output_dir,
        args,
        rows,
        metrics,
        losses,
        metadata["peak_cuda_allocated_gib"],
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "metrics_shape": list(metrics["state_norm"].shape),
        "peak_cuda_allocated_gib": metadata["peak_cuda_allocated_gib"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
