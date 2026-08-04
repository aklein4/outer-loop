"""Rank ungated normalization candidates against the original learned gates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


CANDIDATES = [
    "raw", "old_current", "masked_mean_a", "masked_mean_g",
    "masked_mean_both", "count_scaled_old", "masked_both_preserve_g_eps",
    "sum_both", "g_sum_only",
    "a_masked_mean_only", "g_masked_mean_only", "matrix_rms_raw",
    "matrix_rms_old_current", "matrix_rms_masked_mean_both",
    "feature_mean_both", "feature_mean_a_only", "feature_mean_g_only",
    "joint_mean_both", "joint_mean_a_only", "joint_mean_g_only",
    "joint_l2_both", "a_sequence_g_feature", "a_sequence_g_joint",
    "a_feature_g_sequence", "a_joint_g_sequence",
]

LABELS = {
    "raw": "No normalization",
    "old_current": "Old: mean(a), sum(g)",
    "masked_mean_a": "Masked mean(a), sum(g)",
    "masked_mean_g": "Padded mean(a), masked mean(g)",
    "masked_mean_both": "Masked mean(a, g)",
    "count_scaled_old": "Count-scaled old (eps preserved)",
    "masked_both_preserve_g_eps": "Masked means, preserve g eps",
    "sum_both": "Sum norm(a, g)",
    "g_sum_only": "Sum norm(g) only",
    "a_masked_mean_only": "Masked mean(a) only",
    "g_masked_mean_only": "Masked mean(g) only",
    "matrix_rms_raw": "Final RMS(raw G)",
    "matrix_rms_old_current": "Final RMS(old G)",
    "matrix_rms_masked_mean_both": "Final RMS(mask-mean G)",
    "feature_mean_both": "Feature mean(a, g), dim=-1",
    "feature_mean_a_only": "Feature mean(a) only, dim=-1",
    "feature_mean_g_only": "Feature mean(g) only, dim=-1",
    "joint_mean_both": "Joint mean(a, g), dims=(-2,-1)",
    "joint_mean_a_only": "Joint mean(a) only, dims=(-2,-1)",
    "joint_mean_g_only": "Joint mean(g) only, dims=(-2,-1)",
    "joint_l2_both": "Joint L2(a, g), dims=(-2,-1)",
    "a_sequence_g_feature": "Sequence mean(a), feature mean(g)",
    "a_sequence_g_joint": "Sequence mean(a), joint mean(g)",
    "a_feature_g_sequence": "Feature mean(a), sequence mean(g)",
    "a_joint_g_sequence": "Joint mean(a), sequence mean(g)",
}


def design(data, include_tokens=False):
    layer = data.layer.to_numpy()
    columns = [np.ones(len(data))]
    if include_tokens:
        columns.append(np.log(data.valid_tokens.to_numpy() / 512))
    columns.extend((layer == index).astype(float) for index in range(1, 16))
    return np.column_stack(columns)


def fitted_token_exponent(data, values):
    return float(np.linalg.lstsq(
        design(data, include_tokens=True), np.log(values), rcond=None
    )[0][1])


def summarize(data):
    train = (data.trajectory < 12).to_numpy()
    test = ~train
    target = data.target_norm.to_numpy()
    target_exponent = fitted_token_exponent(data, target)
    records = []
    for candidate in CANDIDATES:
        norm = data[f"{candidate}_norm"].to_numpy()
        cosine = data[f"{candidate}_cos_target"].to_numpy()

        global_scale = np.sum(norm[train] * target[train] * cosine[train]) / np.sum(
            norm[train] ** 2
        )
        global_error = np.sqrt(np.sum(
            (global_scale * norm[test]) ** 2 + target[test] ** 2
            - 2 * global_scale * norm[test] * target[test] * cosine[test]
        ) / np.sum(target[test] ** 2))

        layer_error_square = 0.0
        test_layer = data.layer.to_numpy()[test]
        for layer in range(16):
            fit = train & (data.layer.to_numpy() == layer)
            scale = np.sum(norm[fit] * target[fit] * cosine[fit]) / np.sum(norm[fit] ** 2)
            selected = test_layer == layer
            layer_error_square += np.sum(
                (scale * norm[test][selected]) ** 2 + target[test][selected] ** 2
                - 2 * scale * norm[test][selected] * target[test][selected]
                * cosine[test][selected]
            )
        layer_error = np.sqrt(layer_error_square / np.sum(target[test] ** 2))

        clipped_cosine = np.clip(cosine[test], 0, None)
        direction_error = np.sqrt(np.sum(
            target[test] ** 2 * (1 - clipped_cosine ** 2)
        ) / np.sum(target[test] ** 2))

        log_scale = np.mean(np.log(target[train] / norm[train]))
        global_log_error = np.sqrt(np.mean(
            (np.log(norm[test]) + log_scale - np.log(target[test])) ** 2
        ))
        layer_design = design(data)
        residual = np.log(target) - np.log(norm)
        layer_coefficients = np.linalg.lstsq(
            layer_design[train], residual[train], rcond=None
        )[0]
        layer_log_residual = residual[test] - layer_design[test] @ layer_coefficients
        layer_log_error = np.sqrt(np.mean(layer_log_residual ** 2))

        token_design = design(data, include_tokens=True)
        token_coefficients = np.linalg.lstsq(
            token_design[train], residual[train], rcond=None
        )[0]
        token_log_residual = residual[test] - token_design[test] @ token_coefficients

        fit = linregress(np.log(norm[test]), np.log(target[test]))
        test_layers = data.layer.to_numpy()[test]
        centered_candidate = np.log(norm[test]).copy()
        centered_target = np.log(target[test]).copy()
        for layer in range(16):
            selected = test_layers == layer
            centered_candidate[selected] -= centered_candidate[selected].mean()
            centered_target[selected] -= centered_target[selected].mean()
        within_fit = linregress(centered_candidate, centered_target)
        candidate_exponent = fitted_token_exponent(data, norm)
        records.append({
            "candidate": candidate,
            "label": LABELS[candidate],
            "mean_cosine": cosine[test].mean(),
            "cosine_q10": np.quantile(cosine[test], .1),
            "direction_only_relative_error": direction_error,
            "global_scale_relative_matrix_error": global_error,
            "layer_scale_relative_matrix_error": layer_error,
            "global_scale_log_factor_error": np.exp(global_log_error),
            "layer_scale_log_factor_error": np.exp(layer_log_error),
            "layer_token_scale_log_factor_error": np.exp(np.sqrt(np.mean(token_log_residual ** 2))),
            "candidate_token_exponent": candidate_exponent,
            "target_token_exponent": target_exponent,
            "residual_token_exponent": token_coefficients[1],
            "heldout_log_correlation": fit.rvalue,
            "heldout_log_slope": fit.slope,
            "heldout_within_layer_log_correlation": within_fit.rvalue,
            "heldout_within_layer_log_slope": within_fit.slope,
        })
    return pd.DataFrame(records)


def scatter_grid(data, summary, output):
    train = data.trajectory < 12
    target = data.target_norm.to_numpy()
    colors = plt.get_cmap("tab10").colors
    columns = 4
    rows = int(np.ceil(len(CANDIDATES) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.7 * rows), squeeze=False)
    for index, (candidate, ax) in enumerate(zip(CANDIDATES, axes.flat)):
        norm = data[f"{candidate}_norm"].to_numpy()
        scale = np.exp(np.mean(np.log(target[train] / norm[train])))
        x = scale * norm
        ax.scatter(x, target, s=3, alpha=.08, color=colors[index % 10],
                   edgecolors="none", rasterized=True)
        if x.max() / x.min() < 1.1:
            center = np.sqrt(x.min() * x.max())
            x_lo, x_hi = center / 2, center * 2
        else:
            x_lo, x_hi = x.min() * .8, x.max() * 1.25
        y_lo, y_hi = target.min() * .8, target.max() * 1.25
        identity_lo = max(x_lo, y_lo)
        identity_hi = min(x_hi, y_hi)
        if identity_lo < identity_hi:
            ax.plot([identity_lo, identity_hi], [identity_lo, identity_hi],
                    color="black", linestyle="--", linewidth=1)
        row = summary.loc[summary.candidate == candidate].iloc[0]
        ax.text(.04, .95,
                f"{LABELS[candidate]}\ncos={row.mean_cosine:.2f}, "
                f"matrix err={row.layer_scale_relative_matrix_error:.2f}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.grid(alpha=.15, which="both")
    for ax in axes.flat[len(CANDIDATES):]:
        ax.set_visible(False)
    fig.supxlabel("Candidate norm after training-set global scale")
    fig.supylabel("Original checkpoint post-gate norm")
    fig.suptitle("Original Forte checkpoint: normalization candidates vs learned gated update")
    fig.tight_layout(rect=(.025, .025, 1, .97))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def overview(data, summary, output):
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ordered = summary.sort_values("heldout_within_layer_log_correlation", ascending=False)
    y = np.arange(len(ordered))
    axes[0, 0].barh(y, ordered.heldout_within_layer_log_correlation, color=colors[1])
    axes[0, 0].axvline(1, color="black", linewidth=1, linestyle="--")
    axes[0, 0].set_yticks(y, ordered.label, fontsize=7)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(xlabel="Within-layer corr(log candidate norm, log gated norm)",
                   title="Does gradient-derived magnitude track the learned update?")

    ordered = summary.sort_values("layer_scale_relative_matrix_error")
    y = np.arange(len(ordered))
    axes[0, 1].barh(y, ordered.layer_scale_relative_matrix_error, color=colors[0])
    axes[0, 1].set_yticks(y, ordered.label, fontsize=8)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(xlabel="Held-out relative matrix error",
                   title="Best scalar per layer (magnitude + direction)")

    ordered = summary.sort_values("mean_cosine", ascending=False)
    y = np.arange(len(ordered))
    axes[1, 0].barh(y, ordered.mean_cosine, color=colors[2])
    axes[1, 0].set_yticks(y, ordered.label, fontsize=8)
    axes[1, 0].invert_yaxis(); axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set(xlabel="Mean cosine with learned gated matrix",
                   title="Directional match")

    ordered = summary.sort_values("layer_scale_log_factor_error")
    y = np.arange(len(ordered))
    axes[1, 1].barh(y, ordered.layer_scale_log_factor_error, color=colors[4])
    axes[1, 1].axvline(1, color="black", linewidth=1)
    axes[1, 1].set_yticks(y, ordered.label, fontsize=8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(xlabel="Typical multiplicative magnitude error",
                   title="Magnitude mismatch after one fitted scalar per layer")

    for ax in axes.flat:
        ax.grid(alpha=.2); 
    fig.suptitle("Gradient/update match to the original step-100 gated checkpoint")
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def primitive_scatter(data, output):
    count_scaled = data.old_current_norm * data.valid_tokens / np.sqrt(1024)
    series = [
        ("projected_gradient_norm", data.projected_gradient_norm.to_numpy(),
         "Projected gradient $g$ norm"),
        ("activation_norm", data.activation_norm.to_numpy(), "Activation $a$ norm"),
        ("raw_norm", data.raw_norm.to_numpy(), "Raw $G=g^T a$ norm"),
        ("count_scaled", count_scaled.to_numpy(), "Best count-scaled normalized $G$ norm"),
    ]
    target = data.target_norm.to_numpy()
    train = (data.trajectory < 12).to_numpy()
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    records = []
    for index, (name, values, label) in enumerate(series):
        scale = np.exp(np.mean(np.log(target[train] / values[train])))
        x = scale * values
        fit = linregress(np.log(x), np.log(target))
        ax = axes[index]
        ax.scatter(x, target, s=3, alpha=.08, color=colors[index],
                   edgecolors="none", rasterized=True)
        if x.max() / x.min() < 1.1:
            center = np.sqrt(x.min() * x.max())
            x_lo, x_hi = center / 2, center * 2
        else:
            x_lo, x_hi = x.min() * .8, x.max() * 1.25
        y_lo, y_hi = target.min() * .8, target.max() * 1.25
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set(xlabel=label, title=f"log corr={fit.rvalue:.2f}, slope={fit.slope:.2f}")
        ax.grid(alpha=.15, which="both")
        records.append({"quantity": name, "log_correlation": fit.rvalue,
                        "log_slope": fit.slope, "global_geometric_scale": scale})
    axes[0].set_ylabel("Original checkpoint post-gate update norm")
    fig.suptitle("Raw gradients, activations, and updates versus the learned gated update")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return pd.DataFrame(records)


def epsilon_effect_plot(data, output):
    preserved = data.masked_mean_a_norm * np.sqrt(data.valid_tokens)
    ratio = data.masked_mean_both_norm / preserved
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(data.projected_gradient_norm, ratio, s=3, alpha=.08,
                    color=colors[0], edgecolors="none", rasterized=True)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Raw projected-gradient norm",
                ylabel="Literal mean norm / epsilon-preserved norm",
                title="Epsilon suppresses low-energy gradients")

    axes[1].hist(ratio, bins=80, color=colors[1], alpha=.85)
    axes[1].axvline(np.median(ratio), color="black", linestyle="--",
                    label=f"median={np.median(ratio):.3f}")
    axes[1].set(xlabel="Literal / preserved update norm", ylabel="Observations",
                title="Suppression is not a global constant")
    axes[1].legend()

    layer_values = [ratio[data.layer == layer] for layer in range(16)]
    axes[2].boxplot(layer_values, showfliers=False)
    axes[2].set(xlabel="Layer", ylabel="Literal / preserved update norm",
                title="Epsilon interaction differs by layer")
    axes[2].set_xticks(range(1, 17), range(16))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Changing sum to mean changes the effective gradient epsilon")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    # Exact count scaling implied by converting the old padded-mean/sum rule to
    # valid means while retaining its effective epsilon. The two expressions
    # below differ only in negligible activation-epsilon details.
    data["count_scaled_old_norm"] = (
        data.old_current_norm * data.valid_tokens / np.sqrt(1024)
    )
    data["count_scaled_old_cos_target"] = data.old_current_cos_target
    data["masked_both_preserve_g_eps_norm"] = (
        data.masked_mean_a_norm * np.sqrt(data.valid_tokens)
    )
    data["masked_both_preserve_g_eps_cos_target"] = data.masked_mean_a_cos_target
    summary = summarize(data)
    summary.to_csv(args.output_dir / "normalization_candidate_summary.csv", index=False)
    scatter_grid(data, summary, args.output_dir / "normalization_candidate_scatter_grid.png")
    overview(data, summary, args.output_dir / "normalization_candidate_overview.png")
    primitive = primitive_scatter(data, args.output_dir / "gradient_update_scatter.png")
    primitive.to_csv(args.output_dir / "gradient_update_correlations.csv", index=False)
    epsilon_effect_plot(data, args.output_dir / "epsilon_effect.png")
    print(summary.sort_values("layer_scale_relative_matrix_error").to_string(index=False))


if __name__ == "__main__":
    main()
