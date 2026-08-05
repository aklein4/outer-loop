"""Plot raw-G and negative-applied-update cosines to the previous state."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def average(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=(1, 2))


def framed_limits(*series: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([x[np.isfinite(x)] for x in series])
    low, high = float(values.min()), float(values.max())
    margin = max(0.08 * (high - low), 0.01)
    return max(-1.0, low - margin), min(1.0, high + margin)


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
    output = args.output or args.input_dir / "cos_raw_G_negative_update_vs_previous_state.png"

    data = np.load(args.input_dir / "metrics.npz")
    steps = np.arange(1, data["cos_raw_G_vs_state"].shape[0] + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    raw_g = smooth(average(data["cos_raw_G_vs_state"]), args.smooth_window)
    negative_applied = smooth(
        -average(data["cos_applied_update_vs_state"]), args.smooth_window
    )
    ax.plot(
        steps,
        raw_g,
        linewidth=2.2,
        label="Raw $G$ vs previous state",
    )
    ax.plot(
        steps,
        negative_applied,
        linewidth=2.2,
        label="Negative applied update vs previous state",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    ax.set(
        xlabel="Adaptation step",
        ylabel="Cosine similarity",
        title="Raw $G$ and negative applied update vs previous state",
        ylim=framed_limits(raw_g, negative_applied),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
