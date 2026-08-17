#!/usr/bin/env python3
"""Plot task examples needed to match a same-step LoRA baseline."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src/local_data/icl_results"
DEFAULT_BASELINE = RESULTS_DIR / "baseline_progress/oloop-lora-llama3p2-1b-pre"
DEFAULT_RUN = RESULTS_DIR / "aklein4--horizon-v2_alpha"
DEFAULT_OUTPUT = REPO_ROOT / "figures/icl_step_efficiency.png"


def load_run(run_dir: Path) -> dict[int, tuple[list[int], list[float]]]:
    """Return {training_step: (num_examples, average_loss)}."""
    checkpoints = {}
    for path in sorted(run_dir.glob("*.json")):
        with path.open() as file:
            rows = json.load(file)
        points = sorted(
            (int(row["num_examples"]), float(row["average"])) for row in rows
        )
        checkpoints[int(path.stem)] = (
            [num_examples for num_examples, _ in points],
            [loss for _, loss in points],
        )
    if not checkpoints:
        raise ValueError(f"No JSON checkpoints found in {run_dir}")
    return checkpoints


def loss_at_n(examples: list[int], losses: list[float], target: int) -> float:
    try:
        return losses[examples.index(target)]
    except ValueError as error:
        raise ValueError(f"No result for num_examples={target}") from error


def examples_at_loss(examples: list[int], losses: list[float], target: float) -> float:
    """Find the first loss crossing, interpolating linearly in log(examples + 1)."""
    for index in range(len(examples) - 1):
        left_loss, right_loss = losses[index : index + 2]
        if not min(left_loss, right_loss) <= target <= max(left_loss, right_loss):
            continue
        if target == left_loss or left_loss == right_loss:
            return float(examples[index])
        if target == right_loss:
            return float(examples[index + 1])
        fraction = (target - left_loss) / (right_loss - left_loss)
        return math.expm1(
            math.log1p(examples[index])
            + fraction
            * (math.log1p(examples[index + 1]) - math.log1p(examples[index]))
        )
    return float("nan")


def efficiencies_at_same_step(
    baseline: dict[int, tuple[list[int], list[float]]],
    run: dict[int, tuple[list[int], list[float]]],
) -> dict[int, list[tuple[int, float]]]:
    common_steps = sorted(baseline.keys() & run.keys())
    if not common_steps:
        raise ValueError("The baseline and comparison run have no common checkpoints")

    baseline_levels = set.intersection(
        *(set(examples) for examples, _ in baseline.values())
    )
    run_levels = set.intersection(*(set(examples) for examples, _ in run.values()))
    reference_examples = sorted(
        n for n in baseline_levels & run_levels if n >= 8
    )
    if not reference_examples:
        raise ValueError("The runs have no common num_examples levels >= 8")

    curves = {n: [] for n in reference_examples}
    for step in common_steps:
        baseline_examples, baseline_losses = baseline[step]
        run_examples, run_losses = run[step]
        for n in reference_examples:
            target_loss = loss_at_n(baseline_examples, baseline_losses, n)
            required_examples = examples_at_loss(run_examples, run_losses, target_loss)
            curves[n].append((step, n / required_examples))
    return curves


def make_figure(baseline_dir: Path, run_dir: Path):
    curves = efficiencies_at_same_step(load_run(baseline_dir), load_run(run_dir))
    columns = 4
    rows = math.ceil(len(curves) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(16, 3.6 * rows), sharex=True,
        constrained_layout=True, squeeze=False,
    )
    fig.set_constrained_layout_pads(h_pad=0.08, hspace=0.12)
    label = run_dir.name.removeprefix("aklein4--")
    training_steps = sorted({step for points in curves.values() for step, _ in points})
    for index, (axis, (n, points)) in enumerate(zip(axes.flat, curves.items())):
        axis.plot(
            [step for step, _ in points],
            [efficiency for _, efficiency in points],
            ".-", markersize=9, label=label,
        )
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(FixedLocator(training_steps))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_title(f"× fewer examples to reach LoRA @ {n}")
        if index // columns == rows - 1:
            axis.set_xlabel("Training step (log scale)")
        if index % columns == 0:
            axis.set_ylabel("Sample efficiency")
        axis.tick_params(axis="x", which="both", labelbottom=True)
        axis.tick_params(axis="y", which="both", labelleft=True)
        axis.grid(True, which="both", alpha=0.3)
    for axis in list(axes.flat)[len(curves):]:
        axis.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("Finetuning Sample Efficiency Through Meta-training")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(args.baseline, args.run)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
