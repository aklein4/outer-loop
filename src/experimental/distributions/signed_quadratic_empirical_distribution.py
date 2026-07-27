"""Sample and plot the empirical distribution of the signed quadratic transform."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MU = 0.0
SIGMA = 0.837345664813
N_SAMPLES = 1_000_000
SEED = 20260727


def transform(x: np.ndarray) -> np.ndarray:
    absolute_x = np.abs(x)
    return np.where(
        absolute_x < 1,
        np.sign(x) * x**2,
        2 * np.sign(x) * (absolute_x - 0.5),
    )


rng = np.random.default_rng(SEED)
x = rng.normal(MU, SIGMA, N_SAMPLES)
y_samples = transform(x)

sample_mean = y_samples.mean()
sample_std = y_samples.std()
sample_skew = np.mean((y_samples - sample_mean) ** 3) / sample_std**3

limit = 6
bins = np.linspace(-limit, limit, 301)

fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
ax.hist(
    y_samples,
    bins=bins,
    density=True,
    color="#7b3f98",
    alpha=0.50,
    edgecolor="white",
    linewidth=0.3,
    label=rf"Empirical histogram ({N_SAMPLES:,} samples)",
)

ax.axvline(0, color="0.35", linewidth=0.8, linestyle="--")
ax.text(
    0.98,
    0.72,
    (
        f"sample mean = {sample_mean:.5f}\n"
        f"sample std = {sample_std:.5f}\n"
        f"sample skewness = {sample_skew:.5f}"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
)
ax.set(
    title=rf"Empirical distribution of $Y=f(X)$, $X\sim\mathcal{{N}}(0,{SIGMA:.6f}^2)$",
    xlabel="y",
    ylabel="probability density",
    xlim=(-limit, limit),
    ylim=(0, 2.5),
)
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)

output = Path(__file__).with_name("signed_quadratic_empirical_distribution.png")
fig.savefig(output, dpi=180)
print(f"samples={N_SAMPLES}")
print(f"sample mean={sample_mean:.12f}")
print(f"sample std={sample_std:.12f}")
print(f"sample skewness={sample_skew:.12f}")
print(output)
