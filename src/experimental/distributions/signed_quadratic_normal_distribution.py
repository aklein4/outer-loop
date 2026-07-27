"""Find and plot a Gaussian standardized by a signed quadratic-linear map."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm


def transform(x: np.ndarray) -> np.ndarray:
    """Signed quadratic for |x|<1, continued linearly for |x|>=1."""
    absolute_x = np.abs(x)
    return np.where(
        absolute_x < 1,
        np.sign(x) * x**2,
        2 * np.sign(x) * (absolute_x - 0.5),
    )


def transformed_second_moment(sigma: float) -> float:
    """Exact E[f(X)^2] for X ~ N(0, sigma^2)."""
    a = 1 / sigma
    phi_a = norm.pdf(a)
    upper_tail = ndtr(-a)

    # E[X^4 1{|X|<1}]
    inner = sigma**4 * (
        3 - 2 * ((a**3 + 3 * a) * phi_a + 3 * upper_tail)
    )

    # E[4(|X|-1/2)^2 1{|X|>=1}]
    outer = 8 * (
        sigma**2 * (a * phi_a + upper_tail)
        - sigma * phi_a
        + 0.25 * upper_tail
    )
    return inner + outer


# The transform is odd and strictly increasing, so E[f(X)] = 0 uniquely at mu=0.
mu = 0.0
sigma = brentq(
    lambda candidate: transformed_second_moment(candidate) - 1,
    0.1,
    3.0,
    xtol=1e-14,
)
transformed_mean = 0.0
transformed_std = np.sqrt(transformed_second_moment(sigma))

x = np.linspace(-4.5 * sigma, 4.5 * sigma, 1_400)
px = norm.pdf(x, loc=mu, scale=sigma)

# Exact change-of-variables density. It has an integrable 1/sqrt(|y|)
# singularity at y=0, so zero itself is excluded from the plotting grid.
y_extent = float(transform(np.array([4.5 * sigma]))[0])
negative_y = np.linspace(-y_extent, -1e-4, 1_500)
positive_y = np.linspace(1e-4, y_extent, 1_500)
y = np.concatenate((negative_y, positive_y))
absolute_y = np.abs(y)
inner = absolute_y < 1
inverse = np.where(
    inner,
    np.sign(y) * np.sqrt(absolute_y),
    np.sign(y) * (absolute_y + 1) / 2,
)
jacobian = np.where(inner, 1 / (2 * np.sqrt(absolute_y)), 0.5)
py = norm.pdf(inverse, loc=mu, scale=sigma) * jacobian

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
axes[0].plot(x, px, color="#315a9a", linewidth=2)
axes[0].fill_between(x, px, color="#315a9a", alpha=0.18)
axes[0].axvline(0, color="0.35", linewidth=0.8, linestyle="--")
axes[0].set(
    title=rf"$X\sim\mathcal{{N}}(0,\,{sigma:.6f}^2)$",
    xlabel="x",
    ylabel="density",
)

axes[1].plot(y, py, color="#7b3f98", linewidth=2)
axes[1].fill_between(y, np.minimum(py, 2.5), color="#7b3f98", alpha=0.18)
axes[1].axvline(0, color="0.35", linewidth=0.8, linestyle="--")
axes[1].annotate(
    r"$p_Y(y)\to\infty$ as $y\to0$",
    xy=(0, 2.42),
    xytext=(1.05, 2.12),
    arrowprops={"arrowstyle": "->", "color": "0.25"},
    fontsize=10,
)
axes[1].set(
    title=rf"$Y=f(X)$: mean={transformed_mean:.0f}, std={transformed_std:.6f}",
    xlabel="y",
    ylabel="density",
    xlim=(-y_extent, y_extent),
    ylim=(0, 2.5),
)

for axis in axes:
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)

output = Path(__file__).with_name("signed_quadratic_normal_distribution.png")
fig.savefig(output, dpi=180)
print(f"mu={mu:.12f}")
print(f"sigma={sigma:.12f}")
print(f"transformed mean={transformed_mean:.12f}")
print(f"transformed std={transformed_std:.12f}")
print(output)
