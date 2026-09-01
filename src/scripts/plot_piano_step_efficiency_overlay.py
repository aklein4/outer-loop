#!/usr/bin/env python3
"""Overlay ICL and persona piano sample efficiency through meta-training."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

from plot_icl_step_efficiency import efficiencies_at_same_step, load_run


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA = REPO_ROOT / "src/local_data"
RUN_NAME = "aklein4--horizon-v2_piano-scaled"
BASELINE_NAME = "fresh/oloop-lora-llama3p2-1b-pre"
DEFAULT_ICL_BASELINE = LOCAL_DATA / "icl_results" / BASELINE_NAME
DEFAULT_ICL_RUN = LOCAL_DATA / "icl_results" / RUN_NAME
DEFAULT_PERSONA_BASELINE = LOCAL_DATA / "persona_results" / BASELINE_NAME
DEFAULT_PERSONA_RUN = LOCAL_DATA / "persona_results" / RUN_NAME
DEFAULT_OUTPUT = REPO_ROOT / "figures/piano_step_efficiency_overlay.png"


def make_figure(
    icl_baseline_dir: Path,
    icl_run_dir: Path,
    persona_baseline_dir: Path,
    persona_run_dir: Path,
):
    """Build an efficiency subplot grid with both evaluation datasets."""
    datasets = {
        "ICL": (
            efficiencies_at_same_step(
                load_run(icl_baseline_dir), load_run(icl_run_dir)
            ),
            "blue",
        ),
        "Persona": (
            efficiencies_at_same_step(
                load_run(persona_baseline_dir), load_run(persona_run_dir)
            ),
            "red",
        ),
    }
    example_counts = sorted(set().union(*(curves for curves, _ in datasets.values())))
    columns = 4
    rows = math.ceil(len(example_counts) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 3.6 * rows),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    fig.set_constrained_layout_pads(h_pad=0.08, hspace=0.12)

    training_steps = sorted(
        {
            step
            for curves, _ in datasets.values()
            for points in curves.values()
            for step, _ in points
        }
    )
    for index, (axis, n) in enumerate(zip(axes.flat, example_counts)):
        for label, (curves, color) in datasets.items():
            if n not in curves:
                continue
            points = curves[n]
            axis.plot(
                [step for step, _ in points],
                [efficiency for _, efficiency in points],
                ".-",
                color=color,
                markersize=9,
                label=label,
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

    for axis in list(axes.flat)[len(example_counts) :]:
        axis.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("Piano Sample Efficiency Through Meta-training")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icl-baseline", type=Path, default=DEFAULT_ICL_BASELINE)
    parser.add_argument("--icl-run", type=Path, default=DEFAULT_ICL_RUN)
    parser.add_argument(
        "--persona-baseline", type=Path, default=DEFAULT_PERSONA_BASELINE
    )
    parser.add_argument("--persona-run", type=Path, default=DEFAULT_PERSONA_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(
        args.icl_baseline,
        args.icl_run,
        args.persona_baseline,
        args.persona_run,
    )
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
