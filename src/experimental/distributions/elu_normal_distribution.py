"""Find a Gaussian whose standard ELU transform has mean 0 and variance 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root
from scipy.special import ndtr
from scipy.stats import norm


def elu_moments(mu: float, sigma: float) -> tuple[float, float]:
    """Return E[ELU(X)] and E[ELU(X)^2] for X ~ N(mu, sigma^2)."""
    a = mu / sigma
    positive_probability = ndtr(a)
    negative_probability = ndtr(-a)

    exp1_truncated = np.exp(mu + sigma**2 / 2) * ndtr(
        (-mu - sigma**2) / sigma
    )
    exp2_truncated = np.exp(2 * mu + 2 * sigma**2) * ndtr(
        (-mu - 2 * sigma**2) / sigma
    )

    positive_first = mu * positive_probability + sigma * norm.pdf(a)
    positive_second = (
        (mu**2 + sigma**2) * positive_probability
        + mu * sigma * norm.pdf(a)
    )

    mean = positive_first + exp1_truncated - negative_probability
    second_moment = (
        positive_second
        + exp2_truncated
        - 2 * exp1_truncated
        + negative_probability
    )
    return mean, second_moment


def equations(parameters: np.ndarray) -> np.ndarray:
    """Use log(sigma) so the numerical solve cannot propose sigma <= 0."""
    mu, log_sigma = parameters
    mean, second_moment = elu_moments(mu, np.exp(log_sigma))
    return np.array([mean, second_moment - 1])


solution = root(equations, np.array([-0.2, 0.2]))
if not solution.success:
    raise RuntimeError(solution.message)

mu, log_sigma = solution.x
sigma = np.exp(log_sigma)
elu_mean, elu_second_moment = elu_moments(mu, sigma)
elu_std = np.sqrt(elu_second_moment - elu_mean**2)

# Exact change-of-variables density for Y = ELU(X).
x = np.linspace(mu - 4.5 * sigma, mu + 4.5 * sigma, 1200)
px = norm.pdf(x, loc=mu, scale=sigma)

y = np.linspace(-0.999, max(4.0, mu + 4.5 * sigma), 1600)
inverse = np.where(y <= 0, np.log1p(y), y)
jacobian = np.where(y <= 0, 1 / (1 + y), 1)
py = norm.pdf(inverse, loc=mu, scale=sigma) * jacobian

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
axes[0].plot(x, px, color="#315a9a", linewidth=2)
axes[0].fill_between(x, px, color="#315a9a", alpha=0.18)
axes[0].axvline(0, color="0.35", linewidth=0.8, linestyle="--")
axes[0].set(title=rf"$X\sim\mathcal{{N}}({mu:.6f},\,{sigma:.6f}^2)$", xlabel="x", ylabel="density")

axes[1].plot(y, py, color="#d35f2d", linewidth=2)
axes[1].fill_between(y, py, color="#d35f2d", alpha=0.18)
axes[1].axvline(0, color="0.35", linewidth=0.8, linestyle="--")
axes[1].set(
    title=rf"$Y=\mathrm{{ELU}}(X)$: mean={elu_mean:.2g}, std={elu_std:.6f}",
    xlabel="y",
    ylabel="density",
    xlim=(-1.05, y.max()),
)

for axis in axes:
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)

output = Path(__file__).with_name("elu_normal_distribution.png")
fig.savefig(output, dpi=180)
print(f"mu={mu:.12f}")
print(f"sigma={sigma:.12f}")
print(f"ELU mean={elu_mean:.12e}")
print(f"ELU std={elu_std:.12f}")
print(output)
