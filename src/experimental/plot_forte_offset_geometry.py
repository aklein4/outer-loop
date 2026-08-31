"""Visualize matched-data geometry changes between Forte-offset steps 200 and 250."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


COLORS = {
    "navy": "#24476B",
    "blue": "#3274A1",
    "cyan": "#3AA6B9",
    "orange": "#E1812C",
    "red": "#C44E52",
    "green": "#4C956C",
    "purple": "#8172B2",
    "gray": "#69747C",
    "light": "#EDF2F4",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": .18,
        "grid.linewidth": .7,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def representation_geometry(rep: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Recovery preserves the residual geometry while rebuilding the fast path",
                 fontsize=17, fontweight="bold", x=.065, ha="left")
    fig.text(.065, .93,
             "Matched tokens from the same 4 real 300k-Horizons trajectories · step 200 vs step 250",
             color=COLORS["gray"], fontsize=10.5)

    specs = [
        ("residual_output", "Residual output", COLORS["navy"]),
        ("standard_mlp_write", "Ordinary MLP write", COLORS["green"]),
        ("fast_mlp_write", "Fast MLP write", COLORS["red"]),
    ]
    ax = axes[0, 0]
    for kind, label, color in specs:
        d = rep[rep.kind == kind].sort_values("layer")
        ax.plot(d.layer, d.linear_cka, marker="o", ms=3.5, lw=2.2, color=color, label=label)
    ax.set(title="Relational geometry (linear CKA)", xlabel="Layer", ylabel="CKA",
           xlim=(0, 15), ylim=(.35, 1.015))
    ax.set_xticks(range(0, 16, 2))
    ax.legend(loc="lower left")
    ax.annotate("residual kernel ≈ unchanged", xy=(10, .999), xytext=(7.2, .88),
                arrowprops={"arrowstyle": "->", "color": COLORS["navy"]},
                color=COLORS["navy"], fontweight="bold")

    ax = axes[0, 1]
    for kind, label, color in specs:
        d = rep[rep.kind == kind].sort_values("layer")
        ax.plot(d.layer, d.matched_cosine_mean, marker="o", ms=3.5, lw=2.2,
                color=color, label=label)
    ax.set(title="Same-token direction", xlabel="Layer", ylabel="Mean cosine",
           xlim=(0, 15), ylim=(.35, 1.015))
    ax.set_xticks(range(0, 16, 2))
    ax.annotate("fast write mean = 0.49", xy=(7, rep[(rep.kind == "fast_mlp_write") &
                                                      (rep.layer == 7)].matched_cosine_mean.iloc[0]),
                xytext=(8.1, .64), arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
                color=COLORS["red"], fontweight="bold")

    gate_specs = [
        ("gradient_gate", "Gradient gate", COLORS["purple"]),
        ("offset_gate", "Offset gate", COLORS["orange"]),
        ("fast_output_gradient", "Fast-output gradient", COLORS["cyan"]),
    ]
    ax = axes[1, 0]
    for kind, label, color in gate_specs:
        d = rep[rep.kind == kind].sort_values("layer")
        ax.plot(d.layer, d.linear_cka, marker="o", ms=3.5, lw=2.2, color=color, label=label)
    ax.set(title="Controller/backward geometry also shifts", xlabel="Layer", ylabel="Linear CKA",
           xlim=(0, 15), ylim=(.18, 1.01))
    ax.set_xticks(range(0, 16, 2))
    ax.legend(loc="lower right")
    ax.axvspan(7.5, 13.5, color=COLORS["red"], alpha=.06, lw=0)

    ax = axes[1, 1]
    kinds = ["fast_output_gradient", "normalized_fast_output_gradient"]
    labels = ["Raw fast-output gradient", "RMS-normalized gradient"]
    colors = [COLORS["cyan"], COLORS["blue"]]
    for kind, label, color in zip(kinds, labels, colors):
        d = rep[rep.kind == kind].sort_values("layer")
        ax.plot(d.layer, d.participation_ratio_change, marker="o", ms=3.5,
                lw=2.2, color=color, label=label)
    ax.axhline(1, color="black", lw=1, ls="--", alpha=.7)
    ax.fill_between([-.5, 15.5], 0, 1, color=COLORS["orange"], alpha=.055)
    ax.set(title="Backward signal narrows after recovery", xlabel="Layer",
           ylabel="Participation rank ratio (250 / 200)", xlim=(0, 15), ylim=(.2, 1.55))
    ax.set_xticks(range(0, 16, 2))
    ax.legend(loc="upper right")
    ax.annotate("layers 9–13 retain only\n30–44% of raw rank", xy=(11, .30), xytext=(4.2, .38),
                arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
                color=COLORS["red"], fontweight="bold")

    fig.subplots_adjust(top=.87, hspace=.32, wspace=.23)
    save(fig, output_dir, "geometry_1_representation_pathways")


def recurrent_geometry(rec: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure(figsize=(15, 8.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1], hspace=.34, wspace=.25)
    fig.suptitle("Fast-state geometry diverges progressively over the recurrent trajectory",
                 fontsize=17, fontweight="bold", x=.06, ha="left")
    fig.text(.06, .93, "Averages over 16 layers and 4 matched trajectories; five captured episode positions",
             color=COLORS["gray"], fontsize=10.5)

    ax = fig.add_subplot(gs[0, 0])
    specs = [
        ("raw_G", "Raw gradient G", COLORS["cyan"]),
        ("base_update", "Base update", COLORS["blue"]),
        ("total_update", "Total update", COLORS["purple"]),
        ("state", "Fast state", COLORS["red"]),
        ("grad_buffer", "Gradient buffer", COLORS["green"]),
    ]
    for kind, label, color in specs:
        d = rec[rec.kind == kind].groupby("episode").matrix_cosine.mean()
        ax.plot(d.index, d.values, marker="o", lw=2.25, ms=5, color=color, label=label)
    ax.set(title="The recovered update follows a different direction", xlabel="Episode",
           ylabel="Step 200 ↔ 250 matrix cosine", xlim=(-1, 64), ylim=(.42, .83))
    ax.set_xticks([0, 7, 15, 31, 63])
    ax.legend(ncol=2, loc="lower left")
    ax.annotate("late total update: 0.48 cosine", xy=(63, .479), xytext=(31, .50),
                arrowprops={"arrowstyle": "->", "color": COLORS["purple"]},
                color=COLORS["purple"], fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    for kind, label, color in specs[:4]:
        d = rec[rec.kind == kind].groupby("episode").relative_delta_norm.mean()
        ax.plot(d.index, d.values, marker="o", lw=2.25, ms=5, color=color, label=label)
    ax.axhline(1, color="black", lw=1, ls="--", alpha=.7)
    ax.set(title="Update differences become order-one", xlabel="Episode",
           ylabel="‖step 250 − step 200‖ / ‖step 200‖", xlim=(-1, 64), ylim=(.62, 1.19))
    ax.set_xticks([0, 7, 15, 31, 63])
    ax.text(61.5, 1.01, "difference = old norm", ha="right", va="bottom", fontsize=8.5)

    state = rec[rec.kind == "state"].pivot_table(index="layer", columns="episode",
                                                  values="stable_rank_ratio", aggfunc="mean")
    ax = fig.add_subplot(gs[1, 0])
    cmap = LinearSegmentedColormap.from_list("rank", ["#E9EFF4", "#F6E5C6", COLORS["red"]])
    im = ax.imshow(state.values, aspect="auto", origin="lower", cmap=cmap, vmin=.8, vmax=5.5)
    ax.set(title="Fast states become higher-rank", xlabel="Episode", ylabel="Layer")
    ax.set_xticks(range(len(state.columns)), state.columns)
    ax.set_yticks(range(0, 16, 2), range(0, 16, 2))
    cbar = fig.colorbar(im, ax=ax, pad=.02)
    cbar.set_label("Stable-rank ratio (250 / 200)")
    ax.text(2, 7, "5.4×", ha="center", va="center", color="white", fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    late = rec[(rec.kind == "total_update") & (rec.episode == 63)].groupby("layer").agg(
        cosine=("matrix_cosine", "mean"), norm_ratio=("norm_ratio", "mean"))
    ax.plot(late.index, late.cosine, color=COLORS["purple"], marker="o", lw=2.2,
            label="Update cosine")
    ax.set(title="Depth is reallocated at late recurrence", xlabel="Layer",
           ylabel="Update cosine", xlim=(0, 15), ylim=(.30, .84))
    ax.set_xticks(range(0, 16, 2))
    ax2 = ax.twinx()
    ax2.plot(late.index, late.norm_ratio, color=COLORS["orange"], marker="s", ms=4, lw=2,
             label="Update norm ratio")
    ax2.axhline(1, color="black", lw=1, ls="--", alpha=.55)
    ax2.set_ylabel("Update norm ratio (250 / 200)", color=COLORS["orange"])
    ax2.tick_params(axis="y", colors=COLORS["orange"])
    ax2.set_ylim(.75, 1.48)
    lines = ax.lines + ax2.lines[:1]
    ax.legend(lines, [x.get_label() for x in lines], loc="upper left")

    fig.subplots_adjust(top=.87)
    save(fig, output_dir, "geometry_2_recurrent_state_drift")


def parameter_geometry(param: pd.DataFrame, rep: pd.DataFrame, output_dir: Path) -> None:
    family = param.groupby("family").agg(
        cosine=("weight_cosine", "mean"),
        relative_delta=("relative_delta_norm", "mean"),
        norm_ratio=("norm_ratio", "mean"),
    )
    ordered = [
        "down_fast", "log_lr", "offset_gate_proj", "gradient_gate_proj", "offset_proj",
        "offset_log_lr", "up_fast", "gate_fast", "down_proj", "up_proj", "self_attention",
    ]
    family = family.loc[[x for x in ordered if x in family.index]]
    pretty = {
        "down_fast": "fast read/write map",
        "log_lr": "base log-LR",
        "offset_gate_proj": "offset gate proj.",
        "gradient_gate_proj": "gradient gate proj.",
        "offset_proj": "offset proj.",
        "offset_log_lr": "offset log-LR",
        "up_fast": "fast up map",
        "gate_fast": "fast gate map",
        "down_proj": "ordinary MLP down",
        "up_proj": "ordinary MLP up",
        "self_attention": "self attention",
    }
    labels = [pretty[x] for x in family.index]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), gridspec_kw={"width_ratios": [1.1, 1, 1.05]})
    fig.suptitle("One learned map dominates the parameter-space rotation: down_fast",
                 fontsize=17, fontweight="bold", x=.055, ha="left")
    fig.text(.055, .91,
             "The map that emits the fast residual write also transforms output gradients into G",
             color=COLORS["gray"], fontsize=10.5)

    ax = axes[0]
    y = np.arange(len(family))
    colors = [COLORS["red"] if x == "down_fast" else COLORS["navy"] for x in family.index]
    ax.barh(y, family.cosine, color=colors, height=.65)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(.88, 1.002)
    ax.set_xlabel("Weight-space cosine")
    ax.set_title("Parameter maps")
    ax.axvline(1, color="black", lw=1)
    for yi, val in enumerate(family.cosine):
        ax.text(max(.881, val - .002), yi, f"{val:.3f}", ha="right", va="center",
                color="white" if val < .98 else COLORS["navy"], fontsize=8, fontweight="bold")

    ax = axes[1]
    down = param[param.family == "down_fast"].sort_values("layer")
    ax.plot(down.layer, down.weight_cosine, color=COLORS["red"], marker="o", lw=2.4,
            label="weight cosine")
    ax.set(title="down_fast rotates most in middle layers", xlabel="Layer", ylabel="Cosine",
           xlim=(0, 15), ylim=(.84, 1.005))
    ax.set_xticks(range(0, 16, 2))
    ax2 = ax.twinx()
    ax2.plot(down.layer, down.relative_delta_norm, color=COLORS["orange"], marker="s",
             ms=4, lw=1.9, label="relative displacement")
    ax2.set_ylabel("Relative parameter displacement", color=COLORS["orange"])
    ax2.tick_params(axis="y", colors=COLORS["orange"])
    ax2.set_ylim(0, .68)
    lines = ax.lines + ax2.lines
    ax.legend(lines, [x.get_label() for x in lines], loc="lower right")

    ax = axes[2]
    summary = pd.DataFrame({
        "path": ["Residual output", "Ordinary MLP write", "Fast MLP write", "Output gradient",
                 "Offset gate", "Fast recurrent state"],
        "cosine": [
            rep[rep.kind == "residual_output"].matched_cosine_mean.mean(),
            rep[rep.kind == "standard_mlp_write"].matched_cosine_mean.mean(),
            rep[rep.kind == "fast_mlp_write"].matched_cosine_mean.mean(),
            rep[rep.kind == "fast_output_gradient"].matched_cosine_mean.mean(),
            rep[rep.kind == "offset_gate"].matched_cosine_mean.mean(),
            np.nan,
        ],
    })
    summary.loc[summary.path == "Fast recurrent state", "cosine"] = .64409
    col = [COLORS["navy"], COLORS["green"], COLORS["red"], COLORS["cyan"],
           COLORS["orange"], COLORS["purple"]]
    yy = np.arange(len(summary))
    ax.barh(yy, summary.cosine, color=col, height=.62)
    ax.set_yticks(yy, summary.path)
    ax.invert_yaxis()
    ax.set_xlim(.4, 1.01)
    ax.set_xlabel("Mean same-object cosine")
    ax.set_title("Forward stability hides internal turnover")
    for yi, val in enumerate(summary.cosine):
        ax.text(val - .012, yi, f"{val:.2f}", ha="right", va="center", color="white",
                fontweight="bold", fontsize=9)

    fig.subplots_adjust(top=.83, wspace=.65)
    save(fig, output_dir, "geometry_3_parameter_and_function_map")


def update_branch_geometry(rec: pd.DataFrame, output_dir: Path) -> None:
    branches = rec[rec.kind.isin(["base_update", "offset_update"])].copy()
    summary = branches.groupby("kind").agg(
        cosine=("matrix_cosine", "mean"),
        relative_delta=("relative_delta_norm", "mean"),
        norm_ratio=("norm_ratio", "mean"),
    )
    by_episode = branches.groupby(["kind", "episode"]).agg(
        cosine=("matrix_cosine", "mean"),
        relative_delta=("relative_delta_norm", "mean"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.suptitle("The offset update rotates more, but the gradient update dominates in absolute size",
                 fontsize=17, fontweight="bold", x=.06, ha="left")
    fig.text(.06, .90,
             "Realized post-cap update matrices · offset norm is fixed to 0.25 × gradient-update norm",
             color=COLORS["gray"], fontsize=10.5)

    ax = axes[0]
    x = np.arange(2); width = .34
    values = summary.loc[["base_update", "offset_update"]]
    ax.bar(x - width / 2, values.cosine, width, color=COLORS["blue"],
           label="Step-200 ↔ 250 cosine")
    ax.bar(x + width / 2, values.relative_delta, width, color=COLORS["orange"],
           label="Relative Δ norm")
    ax.set_xticks(x, ["Gradient-based", "Offset-based"])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Layer/episode mean")
    ax.set_title("Offset changes more fractionally")
    ax.legend(loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")

    ax = axes[1]
    styles = [
        ("base_update", "Gradient-based", COLORS["blue"]),
        ("offset_update", "Offset-based", COLORS["orange"]),
    ]
    for kind, label, color in styles:
        d = by_episode.loc[kind]
        ax.plot(d.index, d.cosine, marker="o", lw=2.4, color=color, label=label)
    ax.set(title="Directional change across recurrence", xlabel="Episode",
           ylabel="Step-200 ↔ 250 matrix cosine", xlim=(-1, 64), ylim=(.15, .76))
    ax.set_xticks([0, 7, 15, 31, 63])
    ax.legend(loc="upper right")
    ax.annotate("offset cosine = 0.23", xy=(7, by_episode.loc[("offset_update", 7), "cosine"]),
                xytext=(17, .27), arrowprops={"arrowstyle": "->", "color": COLORS["orange"]},
                color=COLORS["orange"], fontweight="bold")
    ax.text(62, .18, "At equal relative change, the offset contributes\nonly ¼ the absolute norm",
            ha="right", va="bottom", color=COLORS["gray"], fontsize=9)

    fig.subplots_adjust(top=.80, wspace=.26)
    save(fig, output_dir, "geometry_4_base_vs_offset_update")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(
        "src/local_data/horizon_v2_forte_offset_spike_analysis/geometry"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    rep = pd.read_csv(args.data_dir / "representation_geometry.csv")
    rec = pd.read_csv(args.data_dir / "recurrent_matrix_geometry.csv")
    param = pd.read_csv(args.data_dir / "parameter_function_geometry.csv")
    representation_geometry(rep, output_dir)
    recurrent_geometry(rec, output_dir)
    parameter_geometry(param, rep, output_dir)
    update_branch_geometry(rec, output_dir)


if __name__ == "__main__":
    main()
