"""Plot layer-normalized magnitude and layer-averaged cosine trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAGNITUDE_KEYS = {
    "state_norm": "State",
    "raw_G_norm": "Raw $G$",
    "pre_lr_update_norm": "Pre-LR update",
    "applied_update_norm": "Applied update",
}
COSINE_KEYS = {
    "cos_applied_update_vs_state": "Applied update vs state before",
    "cos_pre_lr_update_vs_state": "Pre-LR update vs state before",
    "cos_raw_G_vs_state": "Raw $G$ vs state before",
    "cos_applied_update_vs_state_after": "Applied update vs state after",
}


def layer_normalized_average(values: np.ndarray) -> np.ndarray:
    """Normalize each layer by its global step/trajectory mean, then average."""
    layer_average = np.nanmean(values, axis=(0, 1), keepdims=True)
    normalized = values / np.maximum(layer_average, 1e-30)
    return np.nanmean(normalized, axis=(1, 2))


def layer_average(values: np.ndarray) -> np.ndarray:
    """Average a cosine over trajectories and layers without changing scale."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=(1, 2))


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=float)
    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(sums, counts, out=np.full_like(values, np.nan), where=counts > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("src/local_data/forte_v3_step100_icl_state_probe"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smooth-window", type=int, default=31)
    args = parser.parse_args()
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer")
    output = args.output or args.input_dir / "layer_normalized_averaged_trajectories.png"

    data = np.load(args.input_dir / "metrics.npz")
    steps = np.arange(1, data["state_norm"].shape[0] + 1)

    fig, (magnitude_ax, cosine_ax) = plt.subplots(1, 2, figsize=(15, 5.5))

    for key, label in MAGNITUDE_KEYS.items():
        magnitude_ax.plot(
            steps,
            smooth(layer_normalized_average(data[key]), args.smooth_window),
            linewidth=2,
            label=label,
        )
    magnitude_ax.axhline(1, color="black", linestyle="--", linewidth=1, alpha=0.55)
    magnitude_ax.set(
        xlabel="Adaptation step",
        ylabel="Magnitude / layer global average",
        title="Layer-normalized magnitudes",
    )
    magnitude_ax.grid(alpha=0.25)
    magnitude_ax.legend()

    for key, label in COSINE_KEYS.items():
        cosine_ax.plot(
            steps,
            smooth(layer_average(data[key]), args.smooth_window),
            linewidth=2,
            label=label,
        )
    cosine_ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    cosine_ax.set(
        xlabel="Adaptation step",
        ylabel="Cosine similarity",
        title="Layer-averaged update/state geometry",
        ylim=(-1.0, 1.0),
    )
    cosine_ax.grid(alpha=0.25)
    cosine_ax.legend(fontsize=8)

    fig.suptitle(
        "Forte v3 ICL adaptation: layer-normalized magnitude and averaged cosine trajectories"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
