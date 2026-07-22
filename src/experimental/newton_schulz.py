from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CoefficientSet = tuple[tuple[float, ...], ...]


# Odd scalar polynomials are applied as x <- a*x + b*x**3 + c*x**5.
# For two-term entries, c is treated as zero.
COEFFICIENTS: dict[str, CoefficientSet] = {
    "Standard Muon": (
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
    ),
    "Cesista/You/Jordan per-step": (
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ),
    "Polar Express": (
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        (1.875, -1.25, 0.375),
    ),
}


def muon_scalar_step(x: np.ndarray, coefficients: tuple[float, ...]) -> np.ndarray:
    a, b, *rest = coefficients
    c = rest[0] if rest else 0.0
    x2 = x * x
    return a * x + b * x * x2 + c * x * x2 * x2


def muon_scalar_iteration(x: np.ndarray, coefficients: CoefficientSet) -> np.ndarray:
    y = x.copy()
    for step_coefficients in coefficients:
        y = muon_scalar_step(y, step_coefficients)
    return y


def polar_express_coefficients(steps: int) -> CoefficientSet:
    coefficients = COEFFICIENTS["Polar Express"]
    if steps <= len(coefficients):
        return coefficients[:steps]
    return coefficients + (coefficients[-1],) * (steps - len(coefficients))


def plot_scalar_iterations(output_dir: Path = Path("local_data")) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "newton_schulz_scalar_0_1.png",
        output_dir / "newton_schulz_scalar_0_0p02.png",
        output_dir / "newton_schulz_scalar_0_1_zoom_y.png",
    ]

    ranges = [
        (0.0, 1.0, None, output_paths[0]),
        (0.0, 0.02, None, output_paths[1]),
        (0.0, 1.0, (0.9, 1.1), output_paths[2]),
    ]

    for left, right, ylim, output_path in ranges:
        x = np.linspace(left, right, 4096)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

        for label, coefficients in COEFFICIENTS.items():
            y = muon_scalar_iteration(x, coefficients)
            ax.plot(x, y, label=label, linewidth=2)

        ax.set_xlabel("Input scalar")
        ax.set_ylabel("Output scalar after composed iteration")
        ax.set_title(f"Muon-style scalar Newton-Schulz outputs on [{left:g}, {right:g}]")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    return output_paths


def plot_polar_express_iterations(output_dir: Path = Path("local_data")) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "polar_express_iterations_0_1.png",
        output_dir / "polar_express_iterations_0_0p02.png",
        output_dir / "polar_express_iterations_0_1_zoom_y.png",
    ]
    ranges = [
        (0.0, 1.0, None, output_paths[0]),
        (0.0, 0.02, None, output_paths[1]),
        (0.0, 1.0, (0.9, 1.1), output_paths[2]),
    ]
    iteration_counts = range(1, 7)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(iteration_counts)))

    for left, right, ylim, output_path in ranges:
        x = np.linspace(left, right, 4096)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

        for steps, color in zip(iteration_counts, colors):
            y = muon_scalar_iteration(x, polar_express_coefficients(steps))
            ax.plot(x, y, label=f"{steps} iteration{'s' if steps > 1 else ''}", color=color, linewidth=2)

        ax.set_xlabel("Input scalar")
        ax.set_ylabel("Output scalar after composed iteration")
        ax.set_title(f"Polar Express scalar outputs on [{left:g}, {right:g}]")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(title="Steps")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    return output_paths


def plot_polar_express_10_iterations(output_dir: Path = Path("local_data")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "polar_express_10_iterations_0_1.png"
    x = np.linspace(0.0, 1.0, 4096)
    y = muon_scalar_iteration(x, polar_express_coefficients(10))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot(x, y, color="tab:orange", linewidth=2, label="10 iterations")
    ax.set_xlabel("Input scalar")
    ax.set_ylabel("Output scalar after composed iteration")
    ax.set_title("Polar Express scalar outputs on [0, 1], 10 iterations")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    paths = plot_scalar_iterations() + plot_polar_express_iterations()
    paths.append(plot_polar_express_10_iterations())
    for path in paths:
        print(path)
