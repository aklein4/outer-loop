#!/usr/bin/env python3
"""Reproduce the full six-panel fresh-frozen ICL accuracy figure."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

plt.style.use("tableau-colorblind10")
GRADIENT = plt.get_cmap("viridis_r")
LINE_STYLES = ["-", "--", ":", "-."]

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "src/local_data/icl_acc_results"
DEFAULT_OUTPUT = REPO_ROOT / "figures/acc_plot.png"

REFERENCE_RUN = (
    "fresh_frozen/oloop-lora-llama3p2-1b-pre/base_lr_1e-04.json",
    "LoRA lr=1e-4",
)
RUNS = [
    (
        "aklein4/horizon-v2_piano",
        "Piano",
        [50, 100, 150, 200, 250, 300, 350],
    ),
    (
        "aklein4/horizon-v2_oloop",
        "OLoop",
        [100, 200, 300, 400, 500, 600, 700, 800]
    ),
]

REFERENCE_EXAMPLES = (16, 64, 384)


def resolve_run_path(path: str, step: int | None) -> Path:
    if path.endswith(".json"):
        return DATA_DIR / path
    if step is None:
        raise ValueError(f"A step is required for run directory {path!r}")
    return DATA_DIR / path.replace("/", "--") / f"{step:012d}.json"


def load_scores(path: str, step: int | None, metric: str) -> tuple[list[int], list[float]]:
    with resolve_run_path(path, step).open() as file:
        rows = json.load(file)
    points = sorted((int(row["num_examples"]), float(row[metric])) for row in rows)
    return [x for x, _ in points], [y for _, y in points]


def y_at_x(x: list[int], y: list[float], target: float) -> float:
    """Interpolate y linearly in log(x + 1)."""
    if not x or target < x[0] or target > x[-1]:
        raise ValueError(f"x={target:g} is outside [{x[0]}, {x[-1]}]")
    for index in range(len(x) - 1):
        if x[index] <= target <= x[index + 1]:
            if target == x[index]:
                return y[index]
            if target == x[index + 1]:
                return y[index + 1]
            fraction = (math.log1p(target) - math.log1p(x[index])) / (
                math.log1p(x[index + 1]) - math.log1p(x[index])
            )
            return y[index] + fraction * (y[index + 1] - y[index])
    if len(x) == 1 and target == x[0]:
        return y[0]
    raise ValueError(f"Could not interpolate x={target:g}")


def x_at_y(x: list[int], y: list[float], target: float) -> float:
    """Interpolate x for a target y linearly in log(x + 1)."""
    for index in range(len(x) - 1):
        left, right = y[index], y[index + 1]
        if not min(left, right) <= target <= max(left, right):
            continue
        if target == left or left == right:
            return float(x[index])
        if target == right:
            return float(x[index + 1])
        fraction = (target - left) / (right - left)
        return math.expm1(
            math.log1p(x[index])
            + fraction * (math.log1p(x[index + 1]) - math.log1p(x[index]))
        )
    return float("nan")


def load_runs(metric: str, max_examples: int | None):
    loaded = {}
    configured_runs = [(REFERENCE_RUN[0], REFERENCE_RUN[1], None, "black", "-")]
    for (path, name, steps), line_style in zip(RUNS, LINE_STYLES):
        configured_runs.extend(
            (
                path,
                name,
                step,
                GRADIENT(0.1 + 0.8 * index / max(1, len(steps) - 1)),
                line_style,
            )
            for index, step in enumerate(steps)
        )
    for filename, name, step, color, line_style in configured_runs:
        x, y = load_scores(filename, step, metric)
        points = [(a, b) for a, b in zip(x, y) if max_examples is None or a <= max_examples]
        if not points:
            raise ValueError(f"No points found for {filename}")
        label = name if step is None else f"{name} step={step:03d}"
        loaded[label] = (
            [a for a, _ in points], [b for _, b in points], color, line_style, name, step
        )
    return loaded


def checkpoint_summaries(runs):
    reference_x, reference_y, _, _, _, _ = runs[REFERENCE_RUN[1]]
    targets = {n: y_at_x(reference_x, reference_y, n) for n in REFERENCE_EXAMPLES}
    summaries = {}
    for _, (x, y, _, _, name, step) in runs.items():
        if step is None:
            continue
        efficiencies = {n: n / x_at_y(x, y, target) for n, target in targets.items()}
        summaries.setdefault(name, []).append((step, y_at_x(x, y, 0), efficiencies))
    return summaries


def make_figure(metric: str, max_examples: int | None, ylabel: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.08, hspace=0.08)
    axes = axes.flatten()
    axes[1].sharey(axes[0])
    runs = load_runs(metric, max_examples)

    axes[0].axvline(65, color="black", linestyle="--")
    axes[1].axvline(64, color="black", linestyle="--")
    for label, (x, y, color, line_style, _, _) in runs.items():
        axes[0].plot([value + 1 for value in x], y, marker=".", linestyle=line_style,
                     markersize=10, label=label, color=color)
        axes[1].plot(x, y, marker=".", linestyle=line_style, markersize=10,
                     label=label, color=color)
    axes[0].set_xscale("log")
    axes[0].set_title("Log scale")
    axes[1].set_title("Linear scale")
    axes[1].legend()

    summaries = checkpoint_summaries(runs)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (name, points) in enumerate(summaries.items()):
        points.sort()
        color = cycle[index % len(cycle)]
        axes[2].plot([p[0] for p in points], [p[1] for p in points], ".-",
                     markersize=10, label=name, color=color)
        for axis, reference in zip(axes[3:], REFERENCE_EXAMPLES):
            axis.plot([p[0] for p in points], [p[2][reference] for p in points],
                      ".-", markersize=10, label=name, color=color)

    axes[2].set(xscale="log", title="Zero-shot performance",
                xlabel="Meta-training steps", ylabel=ylabel)
    axes[2].grid(True, which="both", alpha=0.3)
    if summaries:
        axes[2].legend()
    for axis, reference in zip(axes[3:], REFERENCE_EXAMPLES):
        axis.axhline(1, color="black", linestyle="--")
        axis.set(
            xscale="log",
            title=f"Relative sample efficiency\n(versus LoRA @ {reference})",
            xlabel="Meta-training steps",
            ylabel=f"{reference} / # examples to reach loss of LoRA @ {reference}",
        )
        axis.grid(True, which="both", alpha=0.3)
        if summaries:
            axis.legend()
    for axis in axes[:2]:
        axis.set_xlabel("Task examples seen")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric", default="average")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--y-axis", default="Loss (cross-entropy)")
    parser.add_argument("--title", default="Supervised Learning Performance on Bitext Finetuning Datasets")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(args.metric, args.max_steps, args.y_axis, args.title)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)


if __name__ == "__main__":
    main()
