from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
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


MODULE_RE = re.compile(
    r"^(backbone|output|memory)_layers\.layers\.(\d+)\.state_mechanism$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare each Slither chunk state update with the gradient of the "
            "pre-update state from that chunk's LM loss."
        )
    )
    parser.add_argument("--checkpoint", default="aklein4/slither_alpha-350m")
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument("--data-config", default="data/longattn-smollm2.yaml")
    parser.add_argument("--trainer-config", default="trainer/slither-med.yaml")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=constants.LOCAL_DATA_PATH / "slither_state_update_alignment",
    )
    return parser.parse_args()


def load_batch(data_config, num_examples: int, device: torch.device):
    dataset = get_dataset(data_config.dataset.url, data_config.dataset.kwargs)
    iterator = iter(dataset)
    rows = [next(iterator) for _ in range(num_examples)]
    collator = import_collator(data_config.collator.type)(
        **data_config.collator.kwargs
    )
    tokens = collator(rows)["input_ids"].to(device)
    del iterator, dataset, rows
    gc.collect()
    return tokens


def state_mechanisms(model):
    mechanisms = []
    for name, module in model.named_modules():
        match = MODULE_RE.match(name)
        if match is None:
            continue
        family = match.group(1)
        layer = int(match.group(2))
        mechanisms.append(
            {
                "family": family,
                "layer": layer,
                "module": f"{family}:{layer:02d}",
                "mechanism": module,
            }
        )
    family_order = {"backbone": 0, "output": 1, "memory": 2}
    return sorted(mechanisms, key=lambda x: (family_order[x["family"]], x["layer"]))


def chunk_lm_loss(logits, labels, pad_token_id: int):
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).view_as(labels)
    valid = labels != pad_token_id
    per_example = (token_losses * valid).sum(1) / valid.sum(1).clamp_min(1)
    return per_example.mean(), per_example.detach()


def add_alignment_records(
    records: list[dict],
    metadata: dict,
    chunk_idx: int,
    gradient: torch.Tensor,
    update: torch.Tensor,
):
    gradient_flat = gradient.flatten(1).float()
    update_flat = update.flatten(1).float()
    # The expected cosines are small, so accumulate reductions in float64.
    dot = (gradient_flat * update_flat).sum(1, dtype=torch.float64)
    grad_sq_norm = gradient_flat.square().sum(1, dtype=torch.float64)
    update_sq_norm = update_flat.square().sum(1, dtype=torch.float64)
    denominator = (grad_sq_norm * update_sq_norm).sqrt()
    valid = denominator > 0
    raw_cosine = torch.full_like(denominator, torch.nan)
    raw_cosine[valid] = dot[valid] / denominator[valid]

    control_gradient_flat = gradient_flat.roll(shifts=1, dims=0)
    control_dot = (control_gradient_flat * update_flat).sum(
        1, dtype=torch.float64
    )
    control_grad_sq_norm = control_gradient_flat.square().sum(
        1, dtype=torch.float64
    )
    control_denominator = (control_grad_sq_norm * update_sq_norm).sqrt()
    control_valid = control_denominator > 0
    control_raw_cosine = torch.full_like(control_denominator, torch.nan)
    control_raw_cosine[control_valid] = (
        control_dot[control_valid] / control_denominator[control_valid]
    )

    for example in range(gradient.shape[0]):
        records.append(
            {
                "example": example,
                "chunk": chunk_idx,
                "sequence_start": chunk_idx * 1024,
                "family": metadata["family"],
                "layer": metadata["layer"],
                "module": metadata["module"],
                "raw_cosine_gradient_update": float(raw_cosine[example].cpu()),
                "descent_cosine_negative_gradient_update": float(
                    -raw_cosine[example].cpu()
                ),
                "control_raw_cosine_cross_example": float(
                    control_raw_cosine[example].cpu()
                ),
                "control_descent_cosine_cross_example": float(
                    -control_raw_cosine[example].cpu()
                ),
                "dot_gradient_update": float(dot[example].cpu()),
                "control_dot_gradient_update": float(
                    control_dot[example].cpu()
                ),
                "gradient_sq_norm": float(grad_sq_norm[example].cpu()),
                "control_gradient_sq_norm": float(
                    control_grad_sq_norm[example].cpu()
                ),
                "update_sq_norm": float(update_sq_norm[example].cpu()),
                "gradient_norm": float(grad_sq_norm[example].sqrt().cpu()),
                "update_norm": float(update_sq_norm[example].sqrt().cpu()),
                "gradient_is_zero": bool(grad_sq_norm[example] == 0),
                "update_is_zero": bool(update_sq_norm[example] == 0),
            }
        )


def capture_alignment(model, tokens, use_autocast: bool):
    chunk_length = int(model.chunk_length)
    input_chunks = list(tokens[:, :-1].split(chunk_length, dim=1))
    label_chunks = list(tokens[:, 1:].split(chunk_length, dim=1))
    mechanisms = state_mechanisms(model)
    if len(mechanisms) != 24:
        raise RuntimeError(f"Expected 24 state mechanisms, found {len(mechanisms)}")

    # The trainer applies updates for all chunks except the final LM chunk.
    applied_pairs = list(zip(input_chunks, label_chunks))[:-1]
    model.init_state(tokens.shape[0], tokens.device)
    previous_mem = None
    records: list[dict] = []
    losses: list[dict] = []
    started = time.time()

    for chunk_idx, (input_ids, labels) in enumerate(applied_pairs):
        for metadata in mechanisms:
            state = metadata["mechanism"].state
            if state.grad is not None:
                state.grad.zero_()

        if chunk_idx == 0:
            # With mem_states=None, state reads are disabled and dL/dS is undefined.
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                _, new_mem = model(
                    input_ids=input_ids,
                    mem_states=None,
                    skip_logits=True,
                )
            per_example_loss = None
        else:
            with torch.enable_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                logits, new_mem = model(
                    input_ids=input_ids,
                    mem_states=previous_mem,
                )
                loss, per_example_loss = chunk_lm_loss(
                    logits, labels, model.config.pad_token_id
                )
            loss.backward()
            losses.extend(
                {
                    "example": example,
                    "chunk": chunk_idx,
                    "loss": float(per_example_loss[example].cpu()),
                }
                for example in range(tokens.shape[0])
            )

        new_mem = new_mem.detach().float()
        with torch.no_grad():
            for metadata in mechanisms:
                mechanism = metadata["mechanism"]
                update = mechanism.writer(new_mem)
                if chunk_idx > 0:
                    gradient = mechanism.state.grad
                    if gradient is None:
                        raise RuntimeError(
                            f"No state gradient for {metadata['module']} "
                            f"at chunk {chunk_idx}"
                        )
                    add_alignment_records(
                        records, metadata, chunk_idx, gradient, update
                    )
                mechanism.state.add_(update)
                if mechanism.state.grad is not None:
                    mechanism.state.grad.zero_()

        previous_mem = new_mem
        torch.cuda.synchronize()
        print(
            f"chunk {chunk_idx:02d}/{len(applied_pairs) - 1:02d}, "
            f"matched_vectors={len(records)}, elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    model.empty_state()
    return pd.DataFrame(records), pd.DataFrame(losses), mechanisms


def pooled_cosines(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            dot_gradient_update=("dot_gradient_update", "sum"),
            control_dot_gradient_update=("control_dot_gradient_update", "sum"),
            gradient_sq_norm=("gradient_sq_norm", "sum"),
            control_gradient_sq_norm=("control_gradient_sq_norm", "sum"),
            update_sq_norm=("update_sq_norm", "sum"),
        )
    )
    denominator = np.sqrt(grouped["gradient_sq_norm"] * grouped["update_sq_norm"])
    grouped["raw_cosine_gradient_update"] = np.where(
        denominator > 0,
        grouped["dot_gradient_update"] / denominator,
        np.nan,
    )
    grouped["descent_cosine_negative_gradient_update"] = -grouped[
        "raw_cosine_gradient_update"
    ]
    control_denominator = np.sqrt(
        grouped["control_gradient_sq_norm"] * grouped["update_sq_norm"]
    )
    grouped["control_raw_cosine_cross_example"] = np.where(
        control_denominator > 0,
        grouped["control_dot_gradient_update"] / control_denominator,
        np.nan,
    )
    grouped["control_descent_cosine_cross_example"] = -grouped[
        "control_raw_cosine_cross_example"
    ]
    return grouped


def summarize_cosines(
    values: pd.Series,
) -> dict[str, float | int | None]:
    total_count = int(len(values))
    x = values.dropna().to_numpy()
    if x.size == 0:
        return {
            "total_count": total_count,
            "valid_count": 0,
            "undefined_count": total_count,
            "mean": None,
            "std": None,
            "min": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "max": None,
            "fraction_positive": None,
            "fraction_gt_0.05": None,
            "fraction_gt_0.10": None,
            "fraction_lt_minus_0.05": None,
        }
    return {
        "total_count": total_count,
        "valid_count": int(x.size),
        "undefined_count": total_count - int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
        "fraction_positive": float(np.mean(x > 0)),
        "fraction_gt_0.05": float(np.mean(x > 0.05)),
        "fraction_gt_0.10": float(np.mean(x > 0.10)),
        "fraction_lt_minus_0.05": float(np.mean(x < -0.05)),
    }


def plot_cosine_histogram(records: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    ax = axes[0]
    finite = records["descent_cosine_negative_gradient_update"].dropna().to_numpy()
    limit = max(float(np.quantile(np.abs(finite), 0.999)), 5e-4)
    bins = np.linspace(-limit, limit, 81)
    colors = {"backbone": "#4472c4", "output": "#dd8452", "memory": "#55a868"}
    for family in ("backbone", "output", "memory"):
        values = records.loc[
            records.family == family,
            "descent_cosine_negative_gradient_update",
        ].dropna()
        if values.empty:
            continue
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[family],
            label=f"{family} (n={len(values)})",
        )
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        xlabel=r"descent cosine: $\cos(\Delta S_t, -\nabla_{S_t} L_t)$",
        ylabel="density",
        title="Per-mechanism, per-example, per-chunk alignment",
    )
    ax.grid(alpha=0.2)
    ax.legend()

    ax = axes[1]
    ax.hist(
        records["descent_cosine_negative_gradient_update"].dropna(),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="matched example",
    )
    ax.hist(
        records["control_descent_cosine_cross_example"].dropna(),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="cross-example control",
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        xlabel=r"descent cosine: $\cos(\Delta S_t, -\nabla_{S_t} L_t)$",
        title="Matched versus empirical control",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_chunk_trends(
    records: pd.DataFrame, model_pooled: pd.DataFrame, output: Path
):
    family_chunk = (
        records.groupby(["family", "chunk"], as_index=False)[
            "descent_cosine_negative_gradient_update"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    model_chunk = model_pooled.groupby("chunk", as_index=False)[
        "descent_cosine_negative_gradient_update"
    ].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"backbone": "#4472c4", "output": "#dd8452", "memory": "#55a868"}
    for family, frame in family_chunk.groupby("family"):
        if not frame["mean"].notna().any():
            continue
        ax.plot(
            frame["chunk"],
            frame["mean"],
            marker=".",
            color=colors[family],
            label=f"{family}: mean mechanism cosine",
        )
    ax.plot(
        model_chunk["chunk"],
        model_chunk["mean"],
        color="black",
        linewidth=2,
        label="model-wide pooled cosine",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="chunk index",
        ylabel=r"$\cos(\Delta S_t, -\nabla_{S_t} L_t)$",
        title="Gradient-descent alignment by chunk",
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_module_chunk_heatmap(records: pd.DataFrame, output: Path):
    family_order = {"backbone": 0, "output": 1, "memory": 2}
    module_order = (
        records[["family", "layer", "module"]]
        .drop_duplicates()
        .assign(family_rank=lambda x: x.family.map(family_order))
        .sort_values(["family_rank", "layer"])["module"]
        .tolist()
    )
    pivot = (
        records.groupby(["module", "chunk"])[
            "descent_cosine_negative_gradient_update"
        ]
        .mean()
        .unstack("chunk")
        .reindex(module_order)
    )
    limit = max(abs(float(np.nanmin(pivot))), abs(float(np.nanmax(pivot))), 1e-4)
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_yticks(range(len(module_order)), module_order, fontsize=8)
    ax.set(
        xlabel="chunk index",
        ylabel="state mechanism",
        title=r"Mean descent cosine $\cos(\Delta S_t, -\nabla L_t)$",
    )
    fig.colorbar(image, ax=ax, label="descent cosine")
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

    device = torch.device("cuda")
    tokens = load_batch(data_config, args.num_examples, device)
    model = load_checkpoint(
        args.checkpoint,
        args.checkpoint_step,
        attention_kernel="gpu_flash_attention",
    )
    # Match training: fp32 parameters/state and bf16 autocast around model forward.
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records, losses, mechanisms = capture_alignment(
        model, tokens, bool(trainer_config.use_autocast)
    )
    family_pooled = pooled_cosines(records, ["example", "chunk", "family"])
    model_pooled = pooled_cosines(records, ["example", "chunk"])
    directional_derivatives = (
        records.groupby(["example", "chunk"], as_index=False)[
            "dot_gradient_update"
        ]
        .sum()
        .merge(losses, on=["example", "chunk"])
    )
    directional_derivatives["relative_linearized_loss_change"] = (
        directional_derivatives["dot_gradient_update"]
        / directional_derivatives["loss"]
    )

    records.to_csv(output / "alignment_records.csv", index=False)
    losses.to_csv(output / "chunk_losses.csv", index=False)
    family_pooled.to_csv(output / "family_pooled_cosines.csv", index=False)
    model_pooled.to_csv(output / "model_pooled_cosines.csv", index=False)
    directional_derivatives.to_csv(
        output / "model_directional_derivatives.csv", index=False
    )

    module_stats = (
        records.groupby(["family", "layer", "module"])[
            "descent_cosine_negative_gradient_update"
        ]
        .apply(lambda x: pd.Series(summarize_cosines(x)))
        .unstack()
        .reset_index()
    )
    module_stats.to_csv(output / "module_cosine_stats.csv", index=False)

    plot_cosine_histogram(records, output / "cosine_histogram.png")
    plot_chunk_trends(records, model_pooled, output / "chunk_cosine_trends.png")
    plot_module_chunk_heatmap(records, output / "module_chunk_heatmap.png")

    individual_summary = summarize_cosines(
        records["descent_cosine_negative_gradient_update"]
    )
    model_summary = summarize_cosines(
        model_pooled["descent_cosine_negative_gradient_update"]
    )
    individual_control_summary = summarize_cosines(
        records["control_descent_cosine_cross_example"]
    )
    model_control_summary = summarize_cosines(
        model_pooled["control_descent_cosine_cross_example"]
    )
    individual_delta = (
        records["descent_cosine_negative_gradient_update"]
        - records["control_descent_cosine_cross_example"]
    )
    model_delta = (
        model_pooled["descent_cosine_negative_gradient_update"]
        - model_pooled["control_descent_cosine_cross_example"]
    )
    family_summary = {
        family: summarize_cosines(
            family_pooled.loc[
                family_pooled.family == family,
                "descent_cosine_negative_gradient_update",
            ]
        )
        for family in ("backbone", "output", "memory")
    }
    relative_change = directional_derivatives[
        "relative_linearized_loss_change"
    ]
    directional_summary = {
        "mean_gradient_dot_update": float(
            directional_derivatives["dot_gradient_update"].mean()
        ),
        "median_gradient_dot_update": float(
            directional_derivatives["dot_gradient_update"].median()
        ),
        "fraction_negative_gradient_dot_update": float(
            np.mean(directional_derivatives["dot_gradient_update"] < 0)
        ),
        "mean_relative_linearized_loss_change": float(relative_change.mean()),
        "median_relative_linearized_loss_change": float(relative_change.median()),
        "q05_relative_linearized_loss_change": float(
            np.quantile(relative_change, 0.05)
        ),
        "q95_relative_linearized_loss_change": float(
            np.quantile(relative_change, 0.95)
        ),
        "caveat": (
            "This is a first-order hypothetical effect of applying the update "
            "before re-evaluating the same chunk; the actual update is applied "
            "after that chunk and the state path is nonlinear."
        ),
    }
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "dataset": data_config.dataset.url,
        "num_examples": args.num_examples,
        "sequence_length": int(data_config.collator.kwargs.sequence_length),
        "chunk_length": int(model.chunk_length),
        "matched_chunks": list(range(1, 31)),
        "num_state_mechanisms": len(mechanisms),
        "state_matrix_shape": [
            int(model.config.state_size),
            int(model.config.state_size),
        ],
        "comparison": (
            "Update produced after chunk t versus negative gradient of the "
            "pre-update state from only chunk t's mean LM loss."
        ),
        "raw_cosine_note": (
            "Gradient descent corresponds to a negative raw "
            "cosine_gradient_update and a positive descent cosine."
        ),
        "parameter_dtype": "float32",
        "autocast": bool(trainer_config.use_autocast),
        "autocast_dtype": "bfloat16",
        "attention_kernel": "gpu_flash_attention",
        "device": torch.cuda.get_device_name(),
        "individual_mechanism_cosines": individual_summary,
        "individual_cross_example_control_cosines": individual_control_summary,
        "individual_matched_minus_control": {
            **summarize_cosines(individual_delta),
            "fraction_matched_gt_control": float(
                np.mean(individual_delta.dropna() > 0)
            ),
        },
        "model_wide_pooled_cosines": model_summary,
        "model_wide_cross_example_control_cosines": model_control_summary,
        "model_wide_matched_minus_control": {
            **summarize_cosines(model_delta),
            "fraction_matched_gt_control": float(
                np.mean(model_delta.dropna() > 0)
            ),
        },
        "family_pooled_cosines": family_summary,
        "model_directional_derivative": directional_summary,
        "random_isotropic_cosine_std_reference": float(
            1 / math.sqrt(int(model.config.state_size) ** 2)
        ),
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
