"""Analyze which simple defaults explain Forte's learned control behavior."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def read_run(path: Path):
    with (path / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    numeric = {}
    for key in rows[0]:
        if key == "trajectory":
            numeric[key] = np.array([int(r[key]) for r in rows])
        elif key in ("position", "layer", "tokens"):
            numeric[key] = np.array([int(r[key]) for r in rows])
        else:
            numeric[key] = np.array([float(r[key]) for r in rows])
    return numeric, json.loads((path / "metadata.json").read_text())


def design(data, components):
    columns = [np.ones(len(data["tokens"]))]
    names = ["intercept"]
    if "tokens" in components:
        columns.append(np.log(data["tokens"].clip(1)) - math.log(512))
        names.append("log_tokens_over_512")
    if "position" in components:
        p = (data["position"] - 31.5) / 31.5
        columns.extend((p, p * p))
        names.extend(("position_linear", "position_quadratic"))
    if "layer" in components:
        for layer in range(1, 16):
            columns.append((data["layer"] == layer).astype(float))
            names.append(f"layer_{layer}")
    return np.column_stack(columns), names


def heldout_fit(data, response, components, holdout_start=12):
    x, names = design(data, components)
    y = np.log(data[response].clip(1e-30))
    train = data["trajectory"] < holdout_start
    test = ~train
    coef = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
    pred = x[test] @ coef
    baseline = y[train].mean()
    denom = np.square(y[test] - baseline).sum()
    r2 = 1 - np.square(y[test] - pred).sum() / max(denom, 1e-30)
    return {
        "components": list(components),
        "test_log_r2": float(r2),
        "test_log_rmse": float(np.sqrt(np.square(y[test] - pred).mean())),
        "coefficients": {name: float(value) for name, value in zip(names, coef)},
    }


def summarize_behavior(data):
    out = {}
    for key in (
        "activation_gate_mean", "gradient_gate_mean",
        "effective_gate_mean", "effective_gate_geometric_mean",
        "norm_update_over_raw_G", "cos_update_vs_raw_G",
        "activation_gate_lt_0.1", "activation_gate_gt_1.9",
        "gradient_gate_lt_0.1", "gradient_gate_gt_1.9",
    ):
        x = data[key]
        out[key] = {
            "mean": float(x.mean()), "std": float(x.std()),
            "q01": float(np.quantile(x, .01)), "median": float(np.median(x)),
            "q99": float(np.quantile(x, .99)),
        }
    out["valid_tokens"] = {
        "mean": float(data["tokens"].mean()),
        "median": float(np.median(data["tokens"])),
        "min": int(data["tokens"].min()), "max": int(data["tokens"].max()),
    }
    models = {}
    for response in (
        "activation_gate_mean", "gradient_gate_mean",
        "effective_gate_geometric_mean", "norm_update_over_raw_G",
    ):
        models[response] = {
            "+".join(parts) or "constant": heldout_fit(data, response, parts)
            for parts in (
                ("tokens",), ("layer",), ("position",),
                ("tokens", "layer"),
                ("tokens", "layer", "position"),
            )
        }
    out["heldout_fits"] = models
    return out


def clean_state(path):
    raw = torch.load(path, map_location="cpu")
    return {
        ".".join(x for x in key.split(".") if x not in ("_orig_mod", "_module")): value.float()
        for key, value in raw.items()
    }


def control_parameter_analysis(initial_path, trained_path):
    initial = clean_state(initial_path)
    trained = clean_state(trained_path)
    results = {"layers": [], "global": {}}
    delta_energy = total_energy = 0.0
    max_abs = 0.0
    nonfinite = 0
    for key, x0 in initial.items():
        x1 = trained[key]
        d = x1 - x0
        delta_energy += d.square().sum().item()
        total_energy += x0.square().sum().item()
        max_abs = max(max_abs, x1.abs().max().item())
        nonfinite += (~torch.isfinite(x1)).sum().item()
    results["global"] = {
        "relative_parameter_delta": math.sqrt(delta_energy / total_energy),
        "trained_max_abs": max_abs,
        "trained_nonfinite": nonfinite,
    }

    control_keys = [k for k in initial if k.endswith("fast_dynamic_lr.log_lr")]
    for layer, log_key in enumerate(control_keys):
        prefix = log_key.removesuffix("log_lr")
        d = trained[log_key] - initial[log_key]
        f = d.shape[0]
        effective_log_delta = d * math.sqrt(f)
        centered = effective_log_delta - effective_log_delta.mean()
        row = centered.mean(1, keepdim=True)
        col = centered.mean(0, keepdim=True)
        additive = row + col
        additive_fraction = (
            additive.square().sum() / centered.square().sum().clamp_min(1e-30)
        ).item()
        # Randomized low-rank energy is diagnostic only; a fixed seed makes it stable.
        torch.manual_seed(1234 + layer)
        _, singular, _ = torch.svd_lowrank(centered, q=8, niter=2)
        lowrank_fraction = (
            singular.square().sum() / centered.square().sum().clamp_min(1e-30)
        ).item()
        layer_result = {
            "layer": layer,
            "effective_log_lr_delta_mean": effective_log_delta.mean().item(),
            "effective_log_lr_delta_std": effective_log_delta.std(unbiased=False).item(),
            "lr_geometric_multiplier": effective_log_delta.mean().exp().item(),
            "lr_multiplier_q01": torch.quantile(effective_log_delta, .01).exp().item(),
            "lr_multiplier_q99": torch.quantile(effective_log_delta, .99).exp().item(),
            "row_plus_column_fraction_centered_delta": additive_fraction,
            "rank8_fraction_centered_delta": lowrank_fraction,
            "diagonal_log_delta_mean": effective_log_delta.diagonal().mean().item(),
            "offdiagonal_log_delta_mean": (
                (effective_log_delta.sum() - effective_log_delta.diagonal().sum())
                / (effective_log_delta.numel() - f)
            ).item(),
        }
        for gate in ("activation_gate_proj.weight", "gradient_gate_proj.weight"):
            key = prefix + gate
            x0, x1 = initial[key], trained[key]
            delta = x1 - x0
            layer_result[gate] = {
                "relative_delta": (delta.norm() / x0.norm()).item(),
                "initial_trained_cosine": torch.nn.functional.cosine_similarity(
                    x0.flatten(), x1.flatten(), dim=0
                ).item(),
                "delta_mean": delta.mean().item(),
                "delta_std": delta.std(unbiased=False).item(),
            }
        results["layers"].append(layer_result)
    return results


def state_summary(metadata):
    # [position, layer, trajectory] -> summarize relative trajectory shape.
    x = np.asarray(metadata["state_norms"], dtype=float)
    final = x[-1]
    normalized = x / np.maximum(final[None], 1e-30)
    return {
        "mean_final_norm": float(final.mean()),
        "median_final_norm": float(np.median(final)),
        "fraction_of_final_at_positions": {
            str(p): float(normalized[p].mean()) for p in (0, 7, 15, 31, 47, 63)
        },
    }


def make_plot(trained, initial, parameter, trained_meta, initial_meta, output,
              mid=None, mid_meta=None):
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    series = [(initial, "initial", colors[1])]
    if mid is not None:
        series.append((mid, "step 50", colors[2]))
    series.append((trained, "step 100", colors[0]))
    for data, label, color in series:
        tokens = data["tokens"]
        y = data["effective_gate_geometric_mean"]
        edges = np.quantile(tokens, np.linspace(0, 1, 17))
        centers, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            use = (tokens >= lo) & (tokens <= hi)
            centers.append(np.median(tokens[use])); means.append(y[use].mean())
        axes[0, 0].plot(centers, means, marker="o", label=label, color=color)
        layer_means = [y[data["layer"] == layer].mean() for layer in range(16)]
        axes[0, 1].plot(range(16), layer_means, marker="o", label=label, color=color)
    axes[0, 0].set(xlabel="Valid tokens", ylabel="Geometric mean activation × gradient gate",
                   title="Gate default versus episode length")
    axes[0, 1].set(xlabel="Layer", ylabel="Geometric mean effective gate",
                   title="Learned layer schedule")

    lr_mult = [x["lr_geometric_multiplier"] for x in parameter["layers"]]
    axes[1, 0].plot(range(16), lr_mult, marker="o", color=colors[2], label="step 100")
    axes[1, 0].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(xlabel="Layer", ylabel="Step-100 / initial geometric LR",
                   title="Learned static learning-rate multiplier")

    state_series = [(initial_meta, "initial", colors[1])]
    if mid_meta is not None:
        state_series.append((mid_meta, "step 50", colors[2]))
    state_series.append((trained_meta, "step 100", colors[0]))
    for meta, label, color in state_series:
        states = np.asarray(meta["state_norms"], dtype=float)
        normalized = states / np.maximum(states[-1:], 1e-30)
        axes[1, 1].plot(range(64), normalized.mean(axis=(1, 2)), label=label, color=color)
    axes[1, 1].set(xlabel="Episode position", ylabel="State norm / final state norm",
                   title="Average state accumulation shape")
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-run", type=Path, required=True)
    parser.add_argument("--initial-run", type=Path, required=True)
    parser.add_argument("--mid-run", type=Path)
    parser.add_argument("--trained-state", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trained, trained_meta = read_run(args.trained_run)
    initial, initial_meta = read_run(args.initial_run)
    mid, mid_meta = read_run(args.mid_run) if args.mid_run else (None, None)
    parameter = control_parameter_analysis(args.initial_state, args.trained_state)
    report = {
        "trained": summarize_behavior(trained),
        "initial": summarize_behavior(initial),
        "trained_state": state_summary(trained_meta),
        "initial_state": state_summary(initial_meta),
        "parameters": parameter,
    }
    if mid is not None:
        report["mid"] = summarize_behavior(mid)
        report["mid_state"] = state_summary(mid_meta)
    (args.output_dir / "learned_defaults_report.json").write_text(json.dumps(report, indent=2))
    make_plot(trained, initial, parameter, trained_meta, initial_meta,
              args.output_dir / "learned_defaults_overview.png", mid, mid_meta)
    print(json.dumps({
        "report": str((args.output_dir / "learned_defaults_report.json").resolve()),
        "plot": str((args.output_dir / "learned_defaults_overview.png").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
