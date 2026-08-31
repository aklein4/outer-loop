"""Create publication-style visualizations for the Forte-offset spike analysis."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch


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
STEP_COLORS = {150: COLORS["gray"], 200: COLORS["blue"], 250: COLORS["orange"]}
PERIODS = {
    "Pre-spike": (180, 203, "#DCE8F2"),
    "Onset": (204, 206, "#FBE8C5"),
    "Spike": (207, 220, "#F4C7C3"),
    "Recovery": (221, 232, "#FFF0CC"),
    "Recovered": (233, 250, "#DCEEDC"),
}


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .7,
        "legend.frameon": False, "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def shade_periods(ax: plt.Axes, labels: bool = False) -> None:
    ymax = ax.get_ylim()[1]
    for label, (lo, hi, color) in PERIODS.items():
        ax.axvspan(lo - .5, hi + .5, color=color, alpha=.62, lw=0, zorder=0)
        if labels:
            ax.text((lo + hi) / 2, ymax, label, ha="center", va="bottom", fontsize=8,
                    color="#454B50", fontweight="bold")


def spike_chronology(data_dir: Path, output_dir: Path) -> None:
    h = pd.read_csv(data_dir / "wandb_history.csv")
    h = h[h._step.between(180, 250)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.05, 1, .82]})
    fig.suptitle("The gradient/replay failure precedes the visible loss spike", fontsize=17,
                 fontweight="bold", x=.08, ha="left")
    fig.text(.08, .94, "W&B run fyqw3ckb · configured LRs stay fixed · all NaN flags remain zero",
             fontsize=10.5, color=COLORS["gray"])

    ax = axes[0]
    ax.plot(h._step, h.loss, color=COLORS["red"], lw=2.4, label="training objective")
    ax.plot(h._step, h.total_loss, color=COLORS["navy"], lw=1.5, alpha=.85,
            label="mean assistant LM loss")
    ax.set_ylabel("Loss")
    ax.set_ylim(1.25, 5.05)
    shade_periods(ax, labels=True)
    ax.legend(loc="upper left", ncol=2)
    ax.annotate("loss jumps\nstep 207", xy=(207, h.loc[h._step == 207, "loss"].iloc[0]),
                xytext=(198.5, 4.35), arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
                color=COLORS["red"], fontweight="bold")
    ax.annotate("peak loss 4.72\nstep 214", xy=(214, h.loc[h._step == 214, "loss"].iloc[0]),
                xytext=(217.5, 4.55), arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
                color=COLORS["red"], fontweight="bold")

    ax = axes[1]
    ax.semilogy(h._step, h.grad_norm, color=COLORS["purple"], lw=2.4)
    ax.axhline(1, color="black", lw=1, ls="--", alpha=.55, label="post-clip norm = 1")
    ax.set_ylabel("Raw outer-gradient norm\n(log scale)")
    ax.set_ylim(.65, 30000)
    shade_periods(ax)
    ax.annotate("precursor: 1,017\nstep 206", xy=(206, h.loc[h._step == 206, "grad_norm"].iloc[0]),
                xytext=(193.5, 1500), arrowprops={"arrowstyle": "->", "color": COLORS["purple"]},
                color=COLORS["purple"], fontweight="bold")
    ax.annotate("peak: 16,666\nstep 211", xy=(211, h.loc[h._step == 211, "grad_norm"].iloc[0]),
                xytext=(216, 11500), arrowprops={"arrowstyle": "->", "color": COLORS["purple"]},
                color=COLORS["purple"], fontweight="bold")
    ax.legend(loc="lower left")

    ax = axes[2]
    clip = np.minimum(1, 1 / h.grad_norm)
    ax.plot(h._step, h.relative_grad_error, color=COLORS["orange"], lw=2.2,
            label="two-pass replay residual")
    ax.set_ylabel("Replay residual")
    ax.set_ylim(0, .76)
    shade_periods(ax)
    ax2 = ax.twinx()
    ax2.semilogy(h._step, clip, color=COLORS["cyan"], lw=1.7, alpha=.9,
                 label="effective clip multiplier")
    ax2.set_ylabel("Clip multiplier (log)", color=COLORS["cyan"])
    ax2.tick_params(axis="y", colors=COLORS["cyan"])
    ax2.set_ylim(4e-5, .05)
    ax.set_xlabel("Training step")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right", ncol=2)
    ax.annotate("replay error rises\nbefore loss", xy=(206, h.loc[h._step == 206, "relative_grad_error"].iloc[0]),
                xytext=(195, .47), arrowprops={"arrowstyle": "->", "color": COLORS["orange"]},
                color=COLORS["orange"], fontweight="bold")

    fig.subplots_adjust(top=.89, hspace=.14, left=.09, right=.9)
    save(fig, output_dir, "finding_1_spike_chronology")


def episode_position_impact(data_dir: Path, output_dir: Path) -> None:
    e = pd.read_csv(data_dir / "wandb_episode_period_summary.csv")
    lm = e[e.kind == "lm"]
    order = ["pre_180_203", "onset_204_206", "spike_207_220", "recovery_221_232", "post_233_250"]
    labels = {"pre_180_203": "Pre-spike", "onset_204_206": "Onset", "spike_207_220": "Spike",
              "recovery_221_232": "Recovery", "post_233_250": "Recovered"}
    colors = {"pre_180_203": COLORS["blue"], "onset_204_206": COLORS["orange"],
              "spike_207_220": COLORS["red"], "recovery_221_232": COLORS["purple"],
              "post_233_250": COLORS["green"]}
    profiles = {p: lm[lm.period == p].set_index("episode")["mean"] for p in order}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.suptitle("Within-trajectory adaptation remains effective during the spike", fontsize=17,
                 fontweight="bold", x=.07, ha="left")
    fig.text(.07, .91, "The earliest recurrent positions are hit hardest; later positions partially compensate",
             color=COLORS["gray"], fontsize=10.5)
    ax = axes[0]
    for p in order:
        lw = 2.8 if p == "spike_207_220" else 1.8
        ax.plot(profiles[p].index, profiles[p].values, color=colors[p], lw=lw, label=labels[p])
    ax.set(xlabel="Episode position in 64-chunk trajectory", ylabel="Mean assistant LM loss",
           xlim=(0, 63), ylim=(1.15, 8.15))
    ax.legend(ncol=2)

    ax = axes[1]
    ratio = profiles["spike_207_220"] / profiles["pre_180_203"]
    ax.plot(ratio.index, ratio.values, color=COLORS["red"], lw=2.5)
    ax.fill_between(ratio.index, 1, ratio.values, color=COLORS["red"], alpha=.14)
    ax.axhline(1, color="black", lw=1, ls="--")
    early = ratio.iloc[:10].mean(); late = ratio.iloc[60:].mean()
    ax.scatter([4.5, 61.5], [early, late], color=[COLORS["red"], COLORS["green"]], s=70, zorder=5)
    ax.annotate(f"episodes 0–9\n{early:.2f}×", (4.5, early), xytext=(12, 3.45),
                arrowprops={"arrowstyle": "->", "color": COLORS["red"]}, color=COLORS["red"],
                fontweight="bold")
    ax.annotate(f"episodes 60–63\n{late:.2f}×", (61.5, late), xytext=(40, 2.2),
                arrowprops={"arrowstyle": "->", "color": COLORS["green"]}, color=COLORS["green"],
                fontweight="bold")
    ax.set(xlabel="Episode position", ylabel="Spike loss / pre-spike loss", xlim=(0, 63), ylim=(.9, 4.25))
    fig.subplots_adjust(top=.83, wspace=.24)
    save(fig, output_dir, "finding_2_episode_position_impact")


def endpoint_dashboard(data_dir: Path, output_dir: Path) -> None:
    metadata = json.loads((data_dir / "metadata.json").read_text())
    result = {r["step"]: r for r in metadata["checkpoint_results"]}
    p = pd.read_csv(data_dir / "parameter_diffs.csv")
    grads = {s: pd.read_csv(data_dir / f"outer_gradients_step{s}.csv") for s in (200, 250)}

    def optimizer_group(name: str) -> str:
        if "embed_tokens" in name or "lm_head" in name:
            return "frozen"
        if any(k in name for k in ("fast", "embedding_norm", "bidirectional_head", "embedding_state")):
            return "fast"
        return "slow"

    p["optimizer"] = p.name.map(optimizer_group)
    displacement = {}
    for group in ("fast", "slow"):
        x = p[p.optimizer == group]
        pre = np.sum(x.elements * x.delta_150_200_rms ** 2)
        event = np.sum(x.elements * x.delta_200_250_rms ** 2)
        displacement[group] = math.sqrt(event / pre)

    family_rows = []
    for s, d in grads.items():
        d = d[~d.name.str.contains("embed_tokens|lm_head")]
        for family, x in d.groupby("family"):
            family_rows.append({"step": s, "family": family, "norm": np.sqrt(np.square(x.grad_norm).sum())})
    family = pd.DataFrame(family_rows).pivot(index="family", columns="step", values="norm").fillna(0)
    family = family.sort_values(200, ascending=False).head(9)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[.95, 1.25], width_ratios=[1, 1, 1.35], hspace=.4, wspace=.32)
    fig.suptitle("Step 250 is recovered—not a persistently damaged checkpoint", fontsize=18,
                 fontweight="bold", x=.06, ha="left")
    fig.text(.06, .93, "Same four real 300k-Horizons trajectories · complete 64-episode two-pass replay",
             color=COLORS["gray"], fontsize=10.5)

    ax = fig.add_subplot(gs[0, 0])
    losses = np.array([[result[s]["first_mean_lm_loss"], result[s]["first_mean_aux_loss"]] for s in (200, 250)])
    x = np.arange(2); width = .34
    ax.bar(x - width / 2, losses[0], width, color=COLORS["blue"], label="step 200")
    ax.bar(x + width / 2, losses[1], width, color=COLORS["orange"], label="step 250")
    ax.set_xticks(x, ["Assistant LM", "Auxiliary"]); ax.set_ylabel("Mean loss"); ax.set_ylim(1.25, 1.55)
    ax.set_title("Endpoint loss is unchanged")
    ax.legend()
    ax.text(.5, 1.53, "weighted objective −0.09%", ha="center", color=COLORS["green"], fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    vals = [displacement["fast"], displacement["slow"]]
    bars = ax.bar(["Fast optimizer", "Slow optimizer"], vals, color=[COLORS["purple"], COLORS["navy"]], width=.62)
    ax.axhline(1, color="black", ls="--", lw=1)
    ax.set_ylim(.82, 1.03); ax.set_ylabel("RMS displacement ratio\n(200→250) / (150→200)")
    ax.set_title("No oversized parameter move")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + .009, f"{val:.3f}×", ha="center", fontweight="bold")

    ax = fig.add_subplot(gs[0, 2])
    metrics = ["Residual output RMS", "LM-state RMS", "Total update norm", "Final state norm"]
    # Directly measured ratios from the saved tables.
    activations = {s: pd.read_csv(data_dir / f"activation_metrics_step{s}.csv").query("phase=='first'") for s in (200, 250)}
    top = {s: pd.read_csv(data_dir / f"top_activation_metrics_step{s}.csv").query("phase=='first'") for s in (200, 250)}
    dynamic = {s: pd.read_csv(data_dir / f"dynamic_metrics_step{s}.csv").query("phase=='first'") for s in (200, 250)}
    state = {s: pd.read_csv(data_dir / f"state_metrics_step{s}.csv").query("phase=='first' and episode==63") for s in (200, 250)}
    ratios = [
        activations[250].output_rms.mean() / activations[200].output_rms.mean(),
        top[250].query("stage=='lm_states'").rms.mean() / top[200].query("stage=='lm_states'").rms.mean(),
        dynamic[250].total_update_norm.mean() / dynamic[200].total_update_norm.mean(),
        state[250].state_norm.mean() / state[200].state_norm.mean(),
    ]
    y = np.arange(len(metrics))
    ax.barh(y, (np.array(ratios) - 1) * 100,
            color=[COLORS["green"] if r <= 1 else COLORS["orange"] for r in ratios])
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, metrics); ax.invert_yaxis(); ax.set_xlabel("Step 250 vs 200 change (%)")
    ax.set_title("Activation/update scales stay bounded")
    for yi, r in zip(y, ratios):
        val = (r - 1) * 100
        ax.text(val + (.15 if val >= 0 else -.15), yi, f"{val:+.1f}%",
                va="center", ha="left" if val >= 0 else "right", fontweight="bold")
    ax.set_xlim(-7, 4)

    ax = fig.add_subplot(gs[1, :])
    y = np.arange(len(family)); height = .35
    ax.barh(y + height/2, family[200], height, color=COLORS["blue"], label="step 200")
    ax.barh(y - height/2, family[250], height, color=COLORS["orange"], label="step 250")
    ax.set_yticks(y, [x.replace("_", " ") for x in family.index]); ax.invert_yaxis()
    ax.set_xlabel("Same-data outer-gradient norm"); ax.set_title("Post-recovery gradients are broadly smaller")
    ax.legend(ncol=2)
    for yi, (_, row) in enumerate(family.iterrows()):
        ratio = row[250] / row[200]
        ax.text(max(row[200], row[250]) + 2, yi, f"{ratio:.2f}×", va="center", color=COLORS["gray"])
    fig.subplots_adjust(top=.86, left=.09, right=.96, bottom=.08)
    save(fig, output_dir, "finding_3_endpoint_dashboard")


def controller_phase_shift(data_dir: Path, output_dir: Path) -> None:
    dynamic = {s: pd.read_csv(data_dir / f"dynamic_metrics_step{s}.csv").query("phase=='first'").groupby("layer").mean(numeric_only=True)
               for s in (150, 200, 250)}
    states = {s: pd.read_csv(data_dir / f"state_metrics_step{s}.csv").query("phase=='first' and episode==63").set_index("layer")
              for s in (150, 200, 250)}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("The learned fast-update controller shifts from middle to late layers", fontsize=17,
                 fontweight="bold", x=.07, ha="left")
    fig.text(.07, .92, "Step 150 is a trend control; step 200 and 250 are the requested endpoints",
             color=COLORS["gray"], fontsize=10.5)

    specs = [
        ("gradient_gate_mean", "Gradient-gate mean", False, dynamic),
        ("offset_gate_mean", "Offset-gate mean (log scale)", True, dynamic),
        ("total_update_norm", "Total fast-update norm", False, dynamic),
        ("state_norm", "Final recurrent-state norm", False, states),
    ]
    for ax, (metric, title, log, source) in zip(axes.flat, specs):
        for s in (150, 200, 250):
            x = source[s].index
            ax.plot(x, source[s][metric], color=STEP_COLORS[s], lw=2.2 if s != 150 else 1.7,
                    marker="o", ms=3.5, label=f"step {s}")
        ax.axvspan(3.5, 9.5, color=COLORS["blue"], alpha=.07, lw=0)
        ax.axvspan(11.5, 15.5, color=COLORS["orange"], alpha=.08, lw=0)
        if log:
            ax.set_yscale("log")
        ax.set_title(title); ax.set_ylabel(metric.replace("_", " "))
        ax.set_xticks(range(16)); ax.set_xlim(-.3, 15.3)
    for ax in axes[1]: ax.set_xlabel("Fast-weight layer")
    axes[0, 0].legend(ncol=3, loc="upper left")
    handles = [Patch(facecolor=COLORS["blue"], alpha=.12, label="middle layers 4–9"),
               Patch(facecolor=COLORS["orange"], alpha=.14, label="late layers 12–15")]
    axes[1, 1].legend(handles=handles, loc="upper left")
    fig.subplots_adjust(top=.86, hspace=.28, wspace=.2)
    save(fig, output_dir, "finding_4_controller_depth_shift")


def ratio_grid(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> np.ndarray:
    aa = a.pivot(index="layer", columns="episode", values=metric).sort_index()
    bb = b.pivot(index="layer", columns="episode", values=metric).sort_index()
    return np.log2(bb.to_numpy() / aa.to_numpy())


def layer_episode_heatmaps(data_dir: Path, output_dir: Path) -> None:
    dyn = {s: pd.read_csv(data_dir / f"dynamic_metrics_step{s}.csv").query("phase=='first'") for s in (200, 250)}
    state = {s: pd.read_csv(data_dir / f"state_metrics_step{s}.csv").query("phase=='first'") for s in (200, 250)}
    act = {s: pd.read_csv(data_dir / f"activation_metrics_step{s}.csv").query("phase=='first'") for s in (200, 250)}
    panels = [
        (ratio_grid(dyn[200], dyn[250], "gradient_gate_mean"), "Gradient gate"),
        (ratio_grid(dyn[200], dyn[250], "offset_gate_mean"), "Offset gate"),
        (ratio_grid(dyn[200], dyn[250], "total_update_norm"), "Total fast update"),
        (ratio_grid(state[200], state[250], "state_norm"), "Recurrent state"),
        (ratio_grid(dyn[200], dyn[250], "raw_G_norm"), "Raw fast gradient G"),
        (ratio_grid(act[200], act[250], "output_rms"), "Residual output RMS"),
    ]
    lim = max(np.nanquantile(np.abs(x), .985) for x, _ in panels)
    lim = max(lim, .25)
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True, sharey=True)
    fig.suptitle("Where step 250 differs from step 200 across layer and recurrence", fontsize=17,
                 fontweight="bold", x=.075, ha="left")
    fig.text(.075, .93, "Color is log₂(step 250 / step 200): red increases, blue decreases",
             color=COLORS["gray"], fontsize=10.5)
    for ax, (grid, title) in zip(axes.flat, panels):
        im = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest",
                       cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
        ax.set_title(title); ax.set_yticks(range(16)); ax.set_ylabel("Layer")
        ax.axhline(11.5, color="black", lw=.7, ls="--", alpha=.6)
    for ax in axes[-1]: ax.set_xlabel("Episode position")
    cbar = fig.colorbar(im, ax=axes, fraction=.018, pad=.02)
    ticks = np.array([-lim, -lim/2, 0, lim/2, lim])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{2**t:.2f}×" for t in ticks])
    cbar.set_label("Step 250 / step 200")
    fig.subplots_adjust(top=.88, left=.07, right=.91, hspace=.25, wspace=.13)
    save(fig, output_dir, "finding_5_layer_episode_heatmaps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "local_data/horizon_v2_forte_offset_spike_analysis")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    output_dir = args.output_dir or args.data_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    spike_chronology(args.data_dir, output_dir)
    episode_position_impact(args.data_dir, output_dir)
    endpoint_dashboard(args.data_dir, output_dir)
    controller_phase_shift(args.data_dir, output_dir)
    layer_episode_heatmaps(args.data_dir, output_dir)
    manifest = {
        "source_data": str(args.data_dir.resolve()),
        "figures": sorted(p.name for p in output_dir.glob("finding_*.png")),
        "formats": ["png", "svg"],
        "ratio_heatmap_definition": "log2(step250 / step200)",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
