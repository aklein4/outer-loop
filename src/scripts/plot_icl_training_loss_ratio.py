#!/usr/bin/env python3
"""Plot Piano-scaled divided by OLoop-LoRA loss for each N-shot."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from plot_icl_training_loss_difference import load_run


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src/local_data/icl_results"
DEFAULT_PIANO_RUN = RESULTS_DIR / "aklein4--horizon-v2_piano-scaled"
DEFAULT_LORA_RUN = RESULTS_DIR / "fresh/oloop-lora-llama3p2-1b-pre"
DEFAULT_OUTPUT = REPO_ROOT / "figures/icl_training_loss_ratio.png"


def loss_ratios(
    piano: dict[int, list[tuple[int, float]]],
    lora: dict[int, list[tuple[int, float]]],
) -> dict[int, list[tuple[int, float]]]:
    """Return Piano/LoRA loss at common N-shot levels and checkpoints."""
    levels = sorted(piano.keys() & lora.keys())
    if not levels:
        raise ValueError("The runs have no common num_examples levels")

    curves: dict[int, list[tuple[int, float]]] = {}
    for num_examples in levels:
        piano_by_step = dict(piano[num_examples])
        lora_by_step = dict(lora[num_examples])
        common_steps = sorted(piano_by_step.keys() & lora_by_step.keys())
        points = []
        for step in common_steps:
            lora_loss = lora_by_step[step]
            if lora_loss == 0:
                raise ValueError(
                    f"Cannot compute ratio: OLoop-LoRA loss is zero at "
                    f"step={step}, num_examples={num_examples}"
                )
            points.append((step, piano_by_step[step] / lora_loss))
        if points:
            curves[num_examples] = points

    if not curves:
        raise ValueError("The runs have no common checkpoints")
    return curves


def make_figure(piano_dir: Path, lora_dir: Path):
    curves = loss_ratios(load_run(piano_dir), load_run(lora_dir))
    columns = 4
    rows = math.ceil(len(curves) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 3.6 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    for index, (axis, (num_examples, points)) in enumerate(
        zip(axes.flat, curves.items())
    ):
        axis.plot(
            [step for step, _ in points],
            [ratio for _, ratio in points],
            ".-",
            markersize=9,
            label="Piano-scaled / OLoop-LoRA",
        )
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_title(f"{num_examples}-shot")
        if index // columns == rows - 1:
            axis.set_xlabel("Training step (log scale)")
        if index % columns == 0:
            axis.set_ylabel("Average loss ratio")
        axis.tick_params(axis="x", which="both", labelbottom=True)
        axis.tick_params(axis="y", which="both", labelleft=True)
        axis.grid(True, which="both", alpha=0.3)

    for axis in list(axes.flat)[len(curves) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle(
        "Piano-scaled / OLoop-LoRA ICL Loss Through Meta-training\n"
        "Values below 1 favor Piano-scaled"
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piano-run", type=Path, default=DEFAULT_PIANO_RUN)
    parser.add_argument("--lora-run", type=Path, default=DEFAULT_LORA_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(args.piano_run, args.lora_run)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
