#!/usr/bin/env python3
"""Plot LV-Eval assistant loss by horizon position for the two model variants."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "src" / "local_data" / "lv_eval_results"
FIGURES = ROOT / "figures"
MODELS = {
    "Baseline": RESULTS
    / "aklein4--horizon-v2_baseline"
    / "aux=1p0_000000001600.json",
    "Piano-scaled": RESULTS
    / "aklein4--horizon-v2_piano-scaled"
    / "aux=0p1_000000001600.json",
}


def load_series(path: Path) -> tuple[list[int], list[float]]:
    with path.open() as f:
        horizon = json.load(f)["horizon"]

    points = [
        (position, loss)
        for position, (loss, count) in enumerate(
            zip(horizon["assistant_losses"], horizon["counts"], strict=True),
            start=1,
        )
        if count >= 10
    ]
    return [point[0] for point in points], [point[1] for point in points]


def plot(xscale: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path in MODELS.items():
        positions, losses = load_series(path)
        ax.plot(positions, losses, linewidth=2, label=label)

    ax.set_xscale(xscale)
    ax.set_xlabel("Horizon position")
    ax.set_ylabel("Assistant loss")
    ax.set_title(f"LV-Eval assistant loss across horizon ({xscale} x-scale)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    plot("linear", FIGURES / "lv_assistant_loss_horizon_linear.png")
    plot("log", FIGURES / "lv_assistant_loss_horizon_log.png")


if __name__ == "__main__":
    main()
