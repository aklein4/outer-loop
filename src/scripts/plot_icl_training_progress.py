#!/usr/bin/env python3
"""Plot average ICL loss over training for each number of task examples."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src/local_data/icl_results"
DEFAULT_RUNS = (
    RESULTS_DIR / "baseline_progress/oloop-lora-llama3p2-1b-pre",
    RESULTS_DIR / "aklein4--horizon-v2_alpha",
)
DEFAULT_OUTPUT = REPO_ROOT / "figures/icl_training_progress.png"


def run_label(run_dir: Path) -> str:
    if run_dir.parent.name == "baseline_progress":
        return f"{run_dir.name} (baseline)"
    return run_dir.name.removeprefix("aklein4--")


def load_run(run_dir: Path) -> dict[int, list[tuple[int, float]]]:
    """Return {num_examples: [(training_step, average_loss), ...]}."""
    curves: dict[int, list[tuple[int, float]]] = {}
    files = sorted(run_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON checkpoints found in {run_dir}")

    for path in files:
        try:
            step = int(path.stem)
        except ValueError as error:
            raise ValueError(f"Checkpoint filename is not an integer: {path.name}") from error
        with path.open() as file:
            rows = json.load(file)
        for row in rows:
            curves.setdefault(int(row["num_examples"]), []).append(
                (step, float(row["average"]))
            )

    for points in curves.values():
        points.sort()
    return curves


def make_figure(run_dirs: list[Path]):
    runs = [(run_label(path), load_run(path)) for path in run_dirs]
    levels = sorted(set().union(*(curves.keys() for _, curves in runs)))
    columns = 4
    rows = math.ceil(len(levels) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(16, 3.6 * rows), sharex=True,
        constrained_layout=True, squeeze=False,
    )

    for axis, num_examples in zip(axes.flat, levels):
        for label, curves in runs:
            points = curves.get(num_examples, [])
            if points:
                axis.plot(
                    [step for step, _ in points],
                    [loss for _, loss in points],
                    marker="o", label=label,
                )
        axis.set_xscale("log")
        axis.set_title(f"num_examples = {num_examples}")
        axis.grid(True, which="both", alpha=0.3)

    for axis in list(axes.flat)[len(levels):]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Training step (log scale)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Average loss")

    axes.flat[0].legend(fontsize="small")
    fig.suptitle("ICL average loss over training")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, default=list(DEFAULT_RUNS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(args.runs)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
