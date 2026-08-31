#!/usr/bin/env python3
"""Plot persona loss against task examples on log and linear x scales."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable


plt.style.use("tableau-colorblind10")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "src" / "local_data" / "persona_results"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "persona_plot_two_panel.png"
DEFAULT_LORA_LR = 3e-4
LR_PATTERN = re.compile(r"base_lr_(?P<lr>[0-9]+e[-+][0-9]+)")


@dataclass(frozen=True)
class Run:
    path: Path
    label: str
    base_lr: float | None


def format_lr(base_lr: float) -> str:
    """Format a learning rate compactly, e.g. 0.0003 as 3e-4."""
    return f"{base_lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def discover_runs(data_dir: Path) -> list[Run]:
    """Find all persona result files and identify Piano versus LoRA runs."""
    runs = []
    for path in sorted(data_dir.rglob("*.json")):
        relative_path = path.relative_to(data_dir)
        if any("piano" in part.lower() for part in relative_path.parts):
            runs.append(Run(path, "Piano", None))
            continue

        match = LR_PATTERN.search(path.stem)
        base_lr = float(match.group("lr")) if match else DEFAULT_LORA_LR
        runs.append(Run(path, f"LoRA lr={format_lr(base_lr)}", base_lr))

    if not runs:
        raise ValueError(f"No JSON result files found in {data_dir}")
    return runs


def load_scores(path: Path, metric: str) -> tuple[list[int], list[float]]:
    with path.open() as file:
        rows = json.load(file)
    points = sorted((int(row["num_examples"]), float(row[metric])) for row in rows)
    if not points:
        raise ValueError(f"No results found in {path}")
    return [x for x, _ in points], [y for _, y in points]


def make_figure(data_dir: Path, metric: str):
    runs = discover_runs(data_dir)
    lora_lrs = [run.base_lr for run in runs if run.base_lr is not None]
    if not lora_lrs:
        raise ValueError("No LoRA runs found")

    vmin, vmax = min(lora_lrs), max(lora_lrs)
    if vmin == vmax:
        vmin, vmax = vmin / 10, vmax * 10
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis_r")

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True, sharey=True
    )
    for run in runs:
        x, y = load_scores(run.path, metric)
        color = "black" if run.base_lr is None else cmap(norm(run.base_lr))
        axes[0].plot(
            [value + 1 for value in x], y, ".-", markersize=10,
            label=run.label, color=color,
        )
        axes[1].plot(
            x, y, ".-", markersize=10, label=run.label, color=color,
        )

    axes[0].axvline(65, color="black", linestyle="--")
    axes[0].set(xscale="log", yscale="log", title="Log scale", ylabel="Loss (cross-entropy)")
    axes[1].axvline(64, color="black", linestyle="--")
    axes[1].set_title("Linear scale")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Task examples seen")
        axis.grid(True, which="both", alpha=0.3)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.02
    )
    colorbar.set_label("LoRA base learning rate")
    fig.suptitle("Persona Evaluation Performance")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric", default="average")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure(args.data_dir, args.metric)
    print(f"Saving figure to {args.output}")
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)


if __name__ == "__main__":
    main()
