"""Plot applied-update magnitude against the square root of adaptation step.

The normalized panel divides every trajectory's value in a layer by that
layer's mean over all adaptation steps and trajectories.  The traces are
aggregated across trajectories with a median, but are not smoothed over step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def layer_global_average(values: np.ndarray) -> np.ndarray:
    """Return one global step/trajectory average for each layer."""
    return np.nanmean(values, axis=(0, 1))


def normalize_by_layer_global_average(values: np.ndarray) -> np.ndarray:
    """Normalize step x trajectory x layer values by each layer's average."""
    averages = layer_global_average(values)
    return values / np.maximum(averages[None, None, :], 1e-30)


def nan_summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the median and interquartile range over trajectory/layer axes."""
    flattened = values.reshape(values.shape[0], -1)
    return (
        np.nanmedian(flattened, axis=1),
        np.nanpercentile(flattened, 25, axis=1),
        np.nanpercentile(flattened, 75, axis=1),
    )


def plot_panel(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    colors: np.ndarray,
) -> None:
    layer_medians = np.nanmedian(values, axis=1)
    for layer in range(layer_medians.shape[1]):
        ax.plot(
            x,
            layer_medians[:, layer],
            color=colors[layer],
            linewidth=0.9,
            alpha=0.55,
        )

    median, q25, q75 = nan_summary(values)
    ax.plot(x, median, color="black", linewidth=2.0, label="all-layer median")
    ax.fill_between(
        x,
        q25,
        q75,
        color="black",
        alpha=0.12,
        label="25–75%",
    )
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set(
        xlabel=r"$\sqrt{\mathrm{adaptation\ step}}$",
        ylabel=ylabel,
        title=title,
    )
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("src/local_data/forte_v3_step100_icl_state_probe"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.input_dir / "applied_update_magnitude_vs_sqrt_adaptation_step.png"
    values = np.load(args.input_dir / "metrics.npz")["applied_update_norm"]
    steps = np.arange(1, values.shape[0] + 1)
    x = np.sqrt(steps)
    normalized = normalize_by_layer_global_average(values)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, values.shape[2]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    plot_panel(
        axes[0],
        x,
        values,
        "Applied update Frobenius norm",
        "Unsmoothed applied update magnitude",
        colors,
    )
    plot_panel(
        axes[1],
        x,
        normalized,
        "Applied update magnitude / layer global average",
        "Unsmoothed layer-normalized magnitude",
        colors,
    )
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1, alpha=0.55)
    fig.suptitle(
        "Forte v3 ICL adaptation: applied update magnitude vs √(adaptation step)"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
