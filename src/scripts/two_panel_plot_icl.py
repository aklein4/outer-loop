#!/usr/bin/env python3
"""Plot ICL loss against task examples on log and linear scales."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


plt.style.use("tableau-colorblind10")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src" / "local_data" / "icl_results" / "fresh_frozen"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "icl_plot_fresh_frozen_two_panel.png"
RUNS = (
    ("base_lr_1e-05.json", "LoRA lr=1e-5"),
    ("base_lr_3e-05.json", "LoRA lr=3e-5"),
    ("base_lr_1e-04.json", "LoRA lr=1e-4"),
    ("base_lr_3e-04.json", "LoRA lr=3e-4"),
    ("base_lr_1e-03.json", "LoRA lr=1e-3"),
)


def load_scores(filename: str, metric: str) -> tuple[list[int], list[float]]:
    path = DATA_DIR / "oloop-lora-llama3p2-1b-pre" / filename
    with path.open() as file:
        rows = json.load(file)
    points = sorted((int(row["num_examples"]), float(row[metric])) for row in rows)
    return [x for x, _ in points], [y for _, y in points]


def make_figure(metric: str):
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )
    colors = plt.get_cmap("viridis_r")([0.1, 0.28, 0.46, 0.64, 0.82])

    for (filename, label), color in zip(RUNS, colors):
        x, y = load_scores(filename, metric)
        axes[0].plot([value + 1 for value in x], y, ".-", markersize=10,
                     label=label, color=color)
        axes[1].plot(x, y, ".-", markersize=10, label=label, color=color)

    axes[0].axvline(65, color="black", linestyle="--")
    axes[0].set(xscale="log", yscale="log", title="Log scale",
                ylabel="Loss (cross-entropy)")
    axes[1].axvline(64, color="black", linestyle="--")
    axes[1].set_title("Linear scale")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Task examples seen")
        axis.grid(True, which="both", alpha=0.3)

    fig.suptitle("Supervised Learning Performance on Bitext Finetuning Datasets")
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


if __name__ == "__main__":
    main()
