#!/usr/bin/env python3
"""Plot single-projection LoRA ablations on log and linear scales."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


plt.style.use("tableau-colorblind10")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src" / "local_data" / "icl_results" / "single_lora_ablation"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "icl_plot_single_lora_ablation_two_panel.png"
RUNS = (
    ("q_proj.json", "Q projection"),
    ("k_proj.json", "K projection"),
    ("v_proj.json", "V projection"),
    ("o_proj.json", "O projection"),
    ("up_proj.json", "Up projection"),
    ("down_proj.json", "Down projection"),
    ("gate_proj.json", "Gate projection"),
)


def load_scores(filename: str, metric: str) -> tuple[list[int], list[float]]:
    with (DATA_DIR / filename).open() as file:
        rows = json.load(file)
    points = sorted((int(row["num_examples"]), float(row[metric])) for row in rows)
    return [x for x, _ in points], [y for _, y in points]


def plot_run(axes, filename: str, label: str, metric: str, color: str, **kwargs) -> None:
    x, y = load_scores(filename, metric)
    axes[0].plot(
        [value + 1 for value in x], y, ".-", markersize=8,
        label=label, color=color, **kwargs,
    )
    axes[1].plot(
        x, y, ".-", markersize=8, label=label, color=color, **kwargs,
    )


def make_figure(metric: str):
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5), constrained_layout=True
    )

    plot_run(
        axes, "baseline.json", "Baseline", metric, "black",
        linewidth=2.5, zorder=10,
    )
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (filename, label), color in zip(RUNS, palette):
        plot_run(axes, filename, label, metric, color)

    axes[0].axvline(65, color="0.4", linestyle="--")
    axes[0].set(xscale="log", yscale="log", title="Log scale",
                ylabel="Loss (cross-entropy)")
    axes[1].axvline(64, color="0.4", linestyle="--")
    axes[1].set_title("Linear scale")
    axes[1].legend(ncol=2)
    for axis in axes:
        axis.set_xlabel("Task examples seen")
        axis.grid(True, which="both", alpha=0.3)

    fig.suptitle("ICL Performance with a Single LoRA Projection Type")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric", default="average")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(args.metric)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
