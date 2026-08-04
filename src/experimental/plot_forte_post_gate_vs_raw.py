"""Scatter original-rule post-gate G magnitude against raw G magnitude."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fit_log_log(raw: np.ndarray, post: np.ndarray) -> tuple[float, float, float]:
    x = np.log10(raw)
    y = np.log10(post)
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    r2 = 1 - np.square(y - prediction).sum() / np.square(y - y.mean()).sum()
    return float(slope), float(intercept), float(r2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    args = parser.parse_args()

    with args.metrics.open() as handle:
        rows = list(csv.DictReader(handle))
    raw = np.asarray([float(row["raw_G_norm"]) for row in rows])
    post = raw * np.asarray([
        float(row["norm_update_over_raw_G"]) for row in rows
    ])
    layer = np.asarray([int(row["layer"]) for row in rows])

    palette = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(4, 4, figsize=(13, 12), sharex=True, sharey=True)
    fit_rows = []
    x_line = np.geomspace(raw.min(), raw.max(), 256)
    for layer_index, ax in enumerate(axes.flat):
        selected = layer == layer_index
        x = raw[selected]
        y = post[selected]
        slope, intercept, r2 = fit_log_log(x, y)
        fit_rows.append({
            "layer": layer_index,
            "observations": int(selected.sum()),
            "log10_slope": slope,
            "log10_intercept": intercept,
            "log10_r2": r2,
            "linear_pearson": float(np.corrcoef(x, y)[0, 1]),
            "log_pearson": float(np.corrcoef(np.log10(x), np.log10(y))[0, 1]),
        })
        color = palette[layer_index % len(palette)]
        ax.scatter(x, y, s=7, alpha=.20, color=color, edgecolors="none",
                   rasterized=True)
        ax.plot(x_line, 10 ** intercept * x_line ** slope, color="black",
                linewidth=1.3)
        ax.text(.04, .95, f"Layer {layer_index}\nslope={slope:.2f}, $R^2$={r2:.2f}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=.18, which="both")

    overall_slope, overall_intercept, overall_r2 = fit_log_log(raw, post)
    fit_rows.append({
        "layer": "all",
        "observations": len(raw),
        "log10_slope": overall_slope,
        "log10_intercept": overall_intercept,
        "log10_r2": overall_r2,
        "linear_pearson": float(np.corrcoef(raw, post)[0, 1]),
        "log_pearson": float(np.corrcoef(np.log10(raw), np.log10(post))[0, 1]),
    })
    fig.supxlabel(r"Raw $G$ Frobenius norm")
    fig.supylabel(r"Post-gate $G$ Frobenius norm")
    fig.suptitle(
        "Original training rule: post-gate versus raw $G$ magnitude\n"
        f"16 full trajectories, 64 episodes each; overall log-log slope "
        f"{overall_slope:.2f}, $R^2$={overall_r2:.2f}",
        y=.995,
    )
    fig.tight_layout(rect=(.025, .025, 1, .965))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    with args.fits.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0]))
        writer.writeheader()
        writer.writerows(fit_rows)

    print(f"wrote {args.output}")
    print(f"wrote {args.fits}")


if __name__ == "__main__":
    main()
