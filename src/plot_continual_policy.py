"""Plot continual PolicyBench accuracy for Piano and OLoop-LoRA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.constants as constants


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, required=True)
    parser.add_argument("--step", type=int, default=1600)
    parser.add_argument(
        "--boundaries",
        type=float,
        nargs="*",
        default=None,
        help="Task-boundary x coordinates (default: equal chunk boundaries)",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def result_paths(root: Path, step: int):
    label = f"{step:012d}.json"
    return {
        "Piano": root / "fresh" / "piano-llama3p2-1b-pre" / label,
        "LoRA": root / "fresh" / "oloop-lora-llama3p2-1b-pre" / label,
    }


def main():
    args = parse_args()
    if args.n_tasks < 1:
        raise ValueError("--n-tasks must be positive")
    root = Path(constants.LOCAL_DATA_PATH) / f"policy_{args.n_tasks}_results"
    paths = result_paths(root, args.step)
    series = {name: json.loads(path.read_text()) for name, path in paths.items()}

    ncols = 2
    nrows = math.ceil(args.n_tasks / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11, 4.5 * nrows), sharex=True, sharey=True, squeeze=False
    )
    colors = {"Piano": "tab:blue", "LoRA": "tab:orange"}
    max_examples = max(point["num_examples"] for points in series.values() for point in points)
    boundaries = args.boundaries
    if boundaries is None:
        boundaries = [
            max_examples * task_idx / args.n_tasks
            for task_idx in range(1, args.n_tasks)
        ]
    if len(boundaries) != args.n_tasks - 1:
        raise ValueError(
            f"expected {args.n_tasks - 1} --boundaries values, got {len(boundaries)}"
        )

    for task_idx, axis in enumerate(axes.flat, start=1):
        if task_idx > args.n_tasks:
            axis.set_visible(False)
            continue
        task = f"task_{task_idx}"
        for name, points in series.items():
            axis.plot(
                [point["num_examples"] for point in points],
                [point["benchmarks"][task] for point in points],
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=colors[name],
                label=name,
            )
        for boundary in boundaries:
            axis.axvline(boundary, color="black", linestyle="--", linewidth=1, alpha=0.75)
        axis.set_title(task.replace("_", " ").title())
        axis.set_xlim(0, max_examples)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Total examples seen")
        axis.set_ylabel("Accuracy")

    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    output = args.output or Path(constants.REPO_PATH) / "figures" / (
        f"continual_policy_{args.n_tasks}_piano_vs_lora_step_{args.step}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
