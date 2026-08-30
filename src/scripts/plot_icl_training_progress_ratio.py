#!/usr/bin/env python3
"""Plot each N-shot loss as a ratio of zero-shot loss over training."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from plot_icl_training_progress import RESULTS_DIR, load_run, run_label


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = (
    RESULTS_DIR / "fresh/oloop-lora-llama3p2-1b-pre",
    RESULTS_DIR / "aklein4--horizon-v2_piano-scaled",
)
DEFAULT_OUTPUT = REPO_ROOT / "figures/icl_training_progress_ratio.png"


def divide_by_zero_shot(
    curves: dict[int, list[tuple[int, float]]],
) -> dict[int, list[tuple[int, float]]]:
    """Return {num_examples: [(step, loss_n / loss_0), ...]}."""
    if 0 not in curves:
        raise ValueError("Run has no num_examples=0 observations")

    zero_shot_by_step = dict(curves[0])
    ratios: dict[int, list[tuple[int, float]]] = {}
    for num_examples, points in curves.items():
        if num_examples == 0:
            continue
        matched = []
        for step, loss in points:
            if step not in zero_shot_by_step:
                continue
            zero_shot_loss = zero_shot_by_step[step]
            if zero_shot_loss == 0:
                raise ValueError(
                    f"Zero-shot loss is zero at training step {step}"
                )
            matched.append((step, loss / zero_shot_loss))
        if matched:
            ratios[num_examples] = matched

    if not ratios:
        raise ValueError("Run has no nonzero-shot observations paired with zero-shot")
    return ratios


def make_figure(run_dirs: list[Path]):
    runs = [
        (run_label(path), divide_by_zero_shot(load_run(path)))
        for path in run_dirs
    ]
    levels = sorted(set().union(*(curves.keys() for _, curves in runs)))
    columns = 4
    rows = math.ceil(len(levels) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 3.6 * rows),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )

    for axis, num_examples in zip(axes.flat, levels):
        for label, curves in runs:
            points = curves.get(num_examples, [])
            if points:
                axis.plot(
                    [step for step, _ in points],
                    [ratio for _, ratio in points],
                    marker="o",
                    label=label,
                )
        axis.set_xscale("log")
        axis.set_title(f"num_examples = {num_examples}")
        axis.grid(True, which="both", alpha=0.3)

    for axis in list(axes.flat)[len(levels) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Training step (log scale)")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Loss ratio $L(M,n)/L(M,0)$")

    axes.flat[0].legend(fontsize="small")
    fig.suptitle(
        "ICL loss as a fraction of zero-shot loss over training\n"
        "Lower values indicate greater relative improvement from task examples"
    )
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
