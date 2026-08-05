"""Scatter and log-log fit update magnitudes against raw G magnitude."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def log_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    prediction = slope * log_x + intercept
    residual = log_y - prediction
    ss_res = np.square(residual).sum()
    ss_tot = np.square(log_y - log_y.mean()).sum()
    return {
        "observations": int(x.size),
        "log10_slope": float(slope),
        "log10_intercept": float(intercept),
        "log10_r2": float(1.0 - ss_res / max(ss_tot, 1e-30)),
        "log10_pearson": float(np.corrcoef(log_x, log_y)[0, 1]),
    }


def prepare(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[valid], y[valid]


def normalize_by_layer_log_average(values: np.ndarray) -> np.ndarray:
    """Divide each layer by its global geometric mean over steps/trajectories."""
    log_values = np.log10(values.astype(np.float64))
    layer_log_average = np.nanmean(log_values, axis=(0, 1), keepdims=True)
    return values / (10.0 ** layer_log_average)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("src/local_data/forte_v3_step100_icl_state_probe"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input_dir / "update_magnitude_vs_raw_G_scatter.png"

    data = np.load(args.input_dir / "metrics.npz")
    raw = normalize_by_layer_log_average(data["raw_G_norm"])
    metrics = {
        "applied_update_norm": "Applied update / layer geometric mean",
        "pre_lr_update_norm": "Pre-LR update / layer geometric mean",
    }
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, raw.shape[2]))
    fits: dict[str, dict] = {}
    fit_rows: list[dict] = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True)
    for ax, (key, ylabel) in zip(axes, metrics.items()):
        normalized_y = normalize_by_layer_log_average(data[key])
        x, y = prepare(raw, normalized_y)
        fit = log_fit(x, y)
        fits[key] = fit
        for layer in range(raw.shape[2]):
            layer_x, layer_y = prepare(raw[:, :, layer], normalized_y[:, :, layer])
            ax.scatter(
                layer_x,
                layer_y,
                s=3,
                alpha=0.12,
                color=colors[layer],
                edgecolors="none",
                rasterized=True,
            )
            layer_fit = log_fit(layer_x, layer_y)
            fit_rows.append({"metric": key, "layer": layer, **layer_fit})

        line_x = np.geomspace(x.min(), x.max(), 300)
        line_y = 10 ** fit["log10_intercept"] * line_x ** fit["log10_slope"]
        ax.plot(
            line_x,
            line_y,
            color="black",
            linewidth=2,
            label=(
                f"global fit: slope={fit['log10_slope']:.3f}, "
                f"$R^2$={fit['log10_r2']:.3f}"
            ),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Raw $G$ / layer geometric mean")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs raw $G$")
        ax.grid(alpha=0.22, which="both")
        ax.legend()

    fig.suptitle("Forte v3 layer-geometric-mean-normalized update magnitudes versus raw $G$")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    (args.input_dir / "update_magnitude_vs_raw_G_fits.json").write_text(
        json.dumps(fits, indent=2) + "\n"
    )
    with (args.input_dir / "update_magnitude_vs_raw_G_layer_fits.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0]))
        writer.writeheader()
        writer.writerows(fit_rows)
    print(f"wrote {output}")
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
