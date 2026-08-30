#!/usr/bin/env python3
"""Plot Piano-scaled minus OLoop-LoRA loss over training for each N-shot."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src/local_data/icl_results"
DEFAULT_PIANO_RUN = RESULTS_DIR / "aklein4--horizon-v2_piano-scaled"
DEFAULT_LORA_RUN = RESULTS_DIR / "fresh/oloop-lora-llama3p2-1b-pre"
DEFAULT_OUTPUT = REPO_ROOT / "figures/icl_training_loss_difference.png"


def load_run(run_dir: Path) -> dict[int, list[tuple[int, float]]]:
    """Return {num_examples: [(training_step, average_loss), ...]}."""
    curves: dict[int, list[tuple[int, float]]] = {}
    files = sorted(run_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON checkpoints found in {run_dir}")

    for path in files:
        if not path.stem.isdigit():
            continue
        step = int(path.stem)
        with path.open() as file:
            rows = json.load(file)
        for row in rows:
            curves.setdefault(int(row["num_examples"]), []).append(
                (step, float(row["average"]))
            )

    if not curves:
        raise ValueError(f"No numeric JSON checkpoints found in {run_dir}")
    for points in curves.values():
        points.sort()
    return curves


def loss_differences(
    piano: dict[int, list[tuple[int, float]]],
    lora: dict[int, list[tuple[int, float]]],
) -> dict[int, list[tuple[int, float]]]:
    """Return Piano minus LoRA loss at common N-shot levels and checkpoints."""
    levels = sorted(piano.keys() & lora.keys())
    if not levels:
        raise ValueError("The runs have no common num_examples levels")

    curves: dict[int, list[tuple[int, float]]] = {}
    for num_examples in levels:
        piano_by_step = dict(piano[num_examples])
        lora_by_step = dict(lora[num_examples])
        common_steps = sorted(piano_by_step.keys() & lora_by_step.keys())
        if common_steps:
            curves[num_examples] = [
                (step, piano_by_step[step] - lora_by_step[step])
                for step in common_steps
            ]

    if not curves:
        raise ValueError("The runs have no common checkpoints")
    return curves


def make_figure(piano_dir: Path, lora_dir: Path):
    curves = loss_differences(load_run(piano_dir), load_run(lora_dir))
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
            [difference for _, difference in points],
            ".-",
            markersize=9,
            label="Piano-scaled − OLoop-LoRA",
        )
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_title(f"{num_examples}-shot")
        if index // columns == rows - 1:
            axis.set_xlabel("Training step (log scale)")
        if index % columns == 0:
            axis.set_ylabel("Average loss difference")
        axis.tick_params(axis="x", which="both", labelbottom=True)
        axis.tick_params(axis="y", which="both", labelleft=True)
        axis.grid(True, which="both", alpha=0.3)

    for axis in list(axes.flat)[len(curves) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle(
        "Piano-scaled − OLoop-LoRA ICL Loss Through Meta-training\n"
        "Negative values favor Piano-scaled"
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
