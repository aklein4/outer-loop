"""Compare raw, sum-normalized, and proposed mean-normalized Forte G RMS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


def geometric_mean(x: pd.Series) -> float:
    return float(np.exp(np.log(x.clip(lower=1e-30)).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("forte_full_trajectories_step50"))
    args = ap.parse_args()
    metrics = pd.read_csv(args.input_dir / "metrics.csv")
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    fast_size = 1536
    sequence_length = 1024

    metrics["raw_G_rms"] = metrics.raw_G_norm / fast_size
    metrics["sum_ungated_G_rms"] = (
        metrics.raw_G_norm * metrics.norm_normalized_G_over_raw_G / fast_size
    )
    metrics["sum_gated_G_rms"] = (
        metrics.raw_G_norm * metrics.norm_update_over_raw_G / fast_size
    )
    metrics["valid_mean_ungated_G_rms"] = (
        metrics.raw_G_norm * metrics.norm_valid_mean_ungated_over_raw_G / fast_size
    )
    metrics["valid_mean_gated_G_rms"] = (
        metrics.raw_G_norm * metrics.norm_valid_mean_update_over_raw_G / fast_size
    )
    metrics["valid_mean_gated_div_tokens_rms"] = (
        metrics.valid_mean_gated_G_rms / metrics.tokens
    )

    by_position = metrics.groupby("position").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, label in (
        ("raw_G_rms", "Raw G = gᵀa"),
        ("sum_gated_G_rms", "Current gated G (sum normalization)"),
        ("valid_mean_gated_G_rms", "Proposed gated G (masked mean × √n)"),
    ):
        ax.plot(by_position.index, by_position[key], label=label, linewidth=1.8)
    ax.set_yscale("log")
    ax.set(xlabel="Episode position", ylabel="Matrix RMS (log scale)",
           title="Absolute RMS under current and proposed gradient normalization", xlim=(0, 63))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.input_dir / "G_rms_absolute_by_position.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for key, label in (
        ("raw_G_rms", "Raw G = gᵀa"),
        ("sum_ungated_G_rms", "Ungated G (sum normalization)"),
        ("valid_mean_ungated_G_rms", "Ungated G (masked mean × √n)"),
    ):
        ax.plot(by_position.index, by_position[key], label=label, linewidth=1.8)
    ax.set_yscale("log")
    ax.set(xlabel="Episode position", ylabel="Matrix RMS (log scale)",
           title="Ungated normalization comparison", xlim=(0, 63))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.input_dir / "G_rms_ungated_absolute_by_position.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    scales = {}
    for key, label in (
        ("raw_G_rms", "Raw G = gᵀa"),
        ("sum_gated_G_rms", "Current gated G (sum normalization)"),
        ("valid_mean_gated_div_tokens_rms", "Proposed gated G ÷ valid tokens"),
    ):
        values = by_position[key].copy()
        scale = "valid_tokens" if key == "valid_mean_gated_div_tokens_rms" else 1
        center = geometric_mean(values)
        scales[key] = {"explicit_divisor": scale, "geometric_mean": center}
        ax.plot(by_position.index, values / center, label=label, linewidth=1.8)
    ax.axhline(1, color="black", linestyle="--", linewidth=1, alpha=.5)
    ax.set(xlabel="Episode position", ylabel="RMS / trajectory geometric mean",
           title="Comparison-scaled G RMS", xlim=(0, 63))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.input_dir / "G_rms_scaled_by_position.png", dpi=180)
    plt.close(fig)

    selected_layers = (0, 1, 2, 6, 10, 14, 15)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, layer in zip(axes, selected_layers):
        layer_position = metrics[metrics.layer == layer].groupby("position").mean(numeric_only=True)
        for key, label in (
            ("raw_G_rms", "raw"),
            ("sum_gated_G_rms", "sum"),
            ("valid_mean_gated_div_tokens_rms", "masked mean / n"),
        ):
            values = layer_position[key]
            values = values / geometric_mean(values)
            ax.plot(values.index, values, label=label, linewidth=1.2)
        ax.set_title(f"Layer {layer}")
        ax.grid(alpha=.2)
    axes[-1].axis("off")
    axes[0].legend(fontsize=8)
    fig.supxlabel("Episode position")
    fig.supylabel("RMS / layer geometric mean")
    fig.suptitle("Comparison-scaled RMS by layer")
    fig.tight_layout()
    fig.savefig(args.input_dir / "G_rms_scaled_selected_layers.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    length_slopes = {}
    for key, label in (
        ("raw_G_rms", "Raw G"),
        ("sum_ungated_G_rms", "Sum-normalized, ungated"),
        ("sum_gated_G_rms", "Sum-normalized, gated"),
        ("valid_mean_gated_G_rms", "Masked-mean × √n, gated"),
    ):
        fit = linregress(np.log(metrics.tokens), np.log(metrics[key].clip(lower=1e-30)))
        length_slopes[key] = {"log_log_slope": fit.slope, "r_squared": fit.rvalue ** 2}
        bins = pd.qcut(metrics.tokens, 20, duplicates="drop")
        grouped = metrics.assign(_bin=bins).groupby("_bin", observed=True)
        x = grouped.tokens.mean()
        y = grouped[key].apply(geometric_mean)
        y = y / geometric_mean(metrics[key])
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.4,
                label=f"{label} (slope {fit.slope:+.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Valid tokens", ylabel="RMS / geometric mean",
           title="Length scaling of raw and normalized G")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.input_dir / "G_rms_vs_valid_tokens.png", dpi=180)
    plt.close(fig)

    # If gates were compensating the proposed factor while preserving the
    # trained update magnitude, their effective scale would need to be 1/L.
    effective = metrics.effective_gate_geometric_mean
    summary = {
        "checkpoint": metadata["checkpoint"],
        "trajectories": metadata["trajectories"],
        "sequence_length": sequence_length,
        "padded_sequence_length": sequence_length,
        "rms": {
            key: {
                "arithmetic_mean": float(metrics[key].mean()),
                "geometric_mean": geometric_mean(metrics[key]),
                "median": float(metrics[key].median()),
                "q01": float(metrics[key].quantile(.01)),
                "q99": float(metrics[key].quantile(.99)),
            }
            for key in (
                "raw_G_rms", "sum_ungated_G_rms", "sum_gated_G_rms",
                "valid_mean_ungated_G_rms", "valid_mean_gated_G_rms",
            )
        },
        "valid_token_counts": {
            "mean": float(metrics.tokens.mean()),
            "median": float(metrics.tokens.median()),
            "min": int(metrics.tokens.min()),
            "max": int(metrics.tokens.max()),
        },
        "gradient_energy_in_valid_positions": {
            "mean": float(metrics.g_energy_valid_fraction.mean()),
            "median": float(metrics.g_energy_valid_fraction.median()),
            "min": float(metrics.g_energy_valid_fraction.min()),
            "q01": float(metrics.g_energy_valid_fraction.quantile(.01)),
        },
        "effective_gate_geometric_mean": {
            "mean": float(effective.mean()),
            "median": float(effective.median()),
            "q01": float(effective.quantile(.01)),
            "q99": float(effective.quantile(.99)),
            "median_scalar_required_to_cancel_measured_factor": float(
                (1 / metrics.norm_valid_mean_update_over_current_update).median()
            ),
            "median_over_required": float(
                effective.median()
                / (1 / metrics.norm_valid_mean_update_over_current_update).median()
            ),
        },
        "matrix_norm_compensation": {
            "proposed_over_current_mean": float(metrics.norm_valid_mean_update_over_current_update.mean()),
            "proposed_over_current_median": float(metrics.norm_valid_mean_update_over_current_update.median()),
            "proposed_over_current_min": float(metrics.norm_valid_mean_update_over_current_update.min()),
            "proposed_over_current_q01": float(metrics.norm_valid_mean_update_over_current_update.quantile(.01)),
            "proposed_over_current_q99": float(metrics.norm_valid_mean_update_over_current_update.quantile(.99)),
            "proposed_over_current_max": float(metrics.norm_valid_mean_update_over_current_update.max()),
            "ratio_over_valid_token_count_mean": float(
                (metrics.norm_valid_mean_update_over_current_update / metrics.tokens).mean()
            ),
            "ratio_over_valid_token_count_q01": float(
                (metrics.norm_valid_mean_update_over_current_update / metrics.tokens).quantile(.01)
            ),
            "ratio_over_valid_token_count_q99": float(
                (metrics.norm_valid_mean_update_over_current_update / metrics.tokens).quantile(.99)
            ),
        },
        "length_scaling": length_slopes,
        "plot_scaling": scales,
    }
    (args.input_dir / "G_normalization_comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
