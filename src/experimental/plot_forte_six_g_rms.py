"""Plot six raw/normalized G RMS variants against valid-token count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


def combined_rms(values: pd.Series) -> float:
    """RMS over equally sized fast-weight matrices."""
    return float(np.sqrt(np.mean(np.square(values))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("forte_full_trajectory_step100"))
    args = ap.parse_args()
    data = pd.read_csv(args.input_dir / "metrics.csv")
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    fast_size = 1536

    data["raw"] = data.raw_G_norm / fast_size
    data["current"] = data.raw_G_norm * data.norm_normalized_G_over_raw_G / fast_size
    data["current_gated"] = data.raw_G_norm * data.norm_update_over_raw_G / fast_size
    data["masked_a"] = data.raw_G_norm * data.norm_mask_mean_a_only_over_raw_G / fast_size
    data["masked_g"] = data.raw_G_norm * data.norm_mask_mean_g_only_over_raw_G / fast_size
    data["masked_both"] = data.raw_G_norm * data.norm_mask_mean_both_over_raw_G / fast_size

    target_rms = combined_rms(data.current)
    constants = {
        key: target_rms / combined_rms(data[key])
        for key in ("masked_a", "masked_g", "masked_both")
    }
    for key, constant in constants.items():
        data[key] *= constant

    episode_rows = []
    for (trajectory, position), group in data.groupby(["trajectory", "position"]):
        episode_rows.append({
            "trajectory": int(trajectory),
            "position": int(position),
            "valid_tokens": int(group.tokens.iloc[0]),
            **{
                f"{key}_rms": combined_rms(group[key])
                for key in ("raw", "current", "current_gated", "masked_a", "masked_g", "masked_both")
            },
        })
    episodes = pd.DataFrame(episode_rows).sort_values("valid_tokens")
    for key in ("raw", "current", "current_gated", "masked_a", "masked_g", "masked_both"):
        trajectory_average = episodes.groupby("trajectory")[f"{key}_rms"].transform("mean")
        episodes[f"{key}_trajectory_normalized"] = episodes[f"{key}_rms"] / trajectory_average
    episodes.to_csv(args.input_dir / "six_G_rms_by_episode.csv", index=False)

    episodes["token_bin"] = pd.qcut(episodes.valid_tokens, 24, duplicates="drop")
    binned_rows = []
    for token_bin, group in episodes.groupby("token_bin", observed=True):
        row = {
            "valid_tokens_mean": group.valid_tokens.mean(),
            "valid_tokens_min": group.valid_tokens.min(),
            "valid_tokens_max": group.valid_tokens.max(),
            "episodes": len(group),
        }
        for key in ("raw", "current", "current_gated", "masked_a", "masked_g", "masked_both"):
            values = group[f"{key}_rms"]
            row[f"{key}_mean"] = values.mean()
            row[f"{key}_sem"] = values.sem()
            normalized_values = group[f"{key}_trajectory_normalized"]
            row[f"{key}_normalized_mean"] = normalized_values.mean()
            row[f"{key}_normalized_sem"] = normalized_values.sem()
        binned_rows.append(row)
    binned = pd.DataFrame(binned_rows)
    binned.to_csv(args.input_dir / "six_G_rms_binned.csv", index=False)

    labels = {
        "current": "Current normalization",
        "current_gated": "Current normalization + gates",
        "masked_a": "Masked mean on a",
        "masked_g": "Masked mean on g",
        "masked_both": "Masked mean on g and a",
    }
    with plt.style.context("tableau-colorblind10"):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:6]
        fig, ax = plt.subplots(figsize=(11, 7))
        for color, key in zip(colors, labels):
            mean = binned[f"{key}_mean"]
            sem = binned[f"{key}_sem"].fillna(0)
            ax.plot(
                binned.valid_tokens_mean,
                mean,
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=color,
                label=labels[key],
            )
            ax.fill_between(
                binned.valid_tokens_mean,
                mean - sem,
                mean + sem,
                color=color,
                alpha=.12,
                linewidth=0,
            )
        ax.set(
            xlabel="Valid tokens",
            ylabel="G matrix RMS (across 16 layers)",
            title=f"Forte step 100: G normalization averaged over {metadata['trajectories']} full trajectories",
        )
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.input_dir / "six_G_rms_vs_valid_tokens.png", dpi=200)
        plt.close(fig)

        all_labels = {
            "raw": "Raw G",
            "current": "Current normalization",
            "current_gated": "Current normalization + gates",
            "masked_a": "Masked mean on a",
            "masked_g": "Masked mean on g",
            "masked_both": "Masked mean on g and a",
        }
        fig, ax = plt.subplots(figsize=(11, 7))
        for color, key in zip(plt.rcParams["axes.prop_cycle"].by_key()["color"][:6], all_labels):
            mean = binned[f"{key}_normalized_mean"]
            sem = binned[f"{key}_normalized_sem"].fillna(0)
            ax.plot(
                binned.valid_tokens_mean, mean, marker="o", markersize=4,
                linewidth=1.5, color=color, label=all_labels[key],
            )
            ax.fill_between(
                binned.valid_tokens_mean, mean - sem, mean + sem,
                color=color, alpha=.12, linewidth=0,
            )
        ax.axhline(1, color="black", linestyle="--", linewidth=1, alpha=.5)
        ax.set(
            xlabel="Valid tokens",
            ylabel="G matrix RMS / trajectory average RMS",
            title=f"Forte step 100: trajectory-normalized G RMS over {metadata['trajectories']} trajectories",
        )
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.input_dir / "six_G_rms_trajectory_normalized_vs_valid_tokens.png", dpi=200)
        plt.close(fig)

        trajectory_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:metadata["trajectories"]]
        comparisons = {
            "current": "Current normalization",
            "masked_a": "Masked mean on a",
            "masked_g": "Masked mean on g",
            "masked_both": "Masked mean on a and g",
        }
        fits = {}

        def draw_scatter(ax, key, include_trajectory_legend):
            x = episodes.current_gated_trajectory_normalized
            y = episodes[f"{key}_trajectory_normalized"]
            fit = linregress(x, y)
            fits[key] = fit
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            fit_x = np.linspace(lo, hi, 200)
            for color, (trajectory, group) in zip(trajectory_colors, episodes.groupby("trajectory")):
                ax.scatter(
                    group.current_gated_trajectory_normalized,
                    group[f"{key}_trajectory_normalized"],
                    s=20, alpha=.65, color=color,
                    label=f"Trajectory {trajectory}" if include_trajectory_legend else None,
                )
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.1,
                    label="Identity" if include_trajectory_legend else None)
            ax.plot(
                fit_x, fit.intercept + fit.slope * fit_x,
                color="black", linewidth=2,
                label=f"OLS: y={fit.slope:.3f}x{fit.intercept:+.3f}, $R^2$={fit.rvalue**2:.3f}",
            )
            ax.set(
                xlabel="Current normalization + gates: RMS / trajectory average",
                ylabel=f"{comparisons[key]}: RMS / trajectory average",
                title=comparisons[key], xlim=(lo, hi), ylim=(lo, hi),
            )
            ax.grid(alpha=.25)
            ax.legend(ncol=2, fontsize=8)

        for key in comparisons:
            fig, ax = plt.subplots(figsize=(8, 8))
            draw_scatter(ax, key, include_trajectory_legend=True)
            fig.tight_layout()
            fig.savefig(args.input_dir / f"current_gated_vs_{key}_scatter.png", dpi=200)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 13))
        for ax, key in zip(axes.flatten(), comparisons):
            draw_scatter(ax, key, include_trajectory_legend=False)
        fig.suptitle("Normalization variants vs current gated update", fontsize=16)
        fig.tight_layout()
        fig.savefig(args.input_dir / "current_gated_scatter_comparison.png", dpi=200)
        plt.close(fig)

    summary = {
        "checkpoint": metadata["checkpoint"],
        "sources": metadata["sources"],
        "trajectories": metadata["trajectories"],
        "episodes": len(episodes),
        "valid_token_range": [int(episodes.valid_tokens.min()), int(episodes.valid_tokens.max())],
        "global_scale_target": "RMS of current ungated normalization over all 64x16 matrices",
        "current_target_rms": target_rms,
        "global_scaling_constants": constants,
        "formulas": {
            "raw": "g.T @ a",
            "current": "g_sum_norm.T @ a_padded_mean_norm",
            "current_gated": "(gradient_gate*g_sum_norm).T @ (activation_gate*a_padded_mean_norm)",
            "masked_a": "global_constant * g_sum_norm.T @ a_masked_mean_norm",
            "masked_g": "global_constant * g_masked_mean_norm.T @ a_padded_mean_norm",
            "masked_both": "global_constant * g_masked_mean_norm.T @ a_masked_mean_norm",
        },
        "palette": "matplotlib tableau-colorblind10",
        "series_global_rms_after_scaling": {
            key: combined_rms(data[key]) for key in labels
        },
        "current_gated_scatter_fits": {
            key: {
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r_squared": fit.rvalue ** 2,
                "p_value": fit.pvalue,
                "standard_error": fit.stderr,
            }
            for key, fit in fits.items()
        },
    }
    (args.input_dir / "six_G_rms_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
