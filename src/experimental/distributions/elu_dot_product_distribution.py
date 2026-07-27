"""Compare 1024-D ELU-Gaussian dot products with Gaussian dot products."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln, logsumexp


DIMENSION = 1024
N_SAMPLES = 300_000
BATCH_SIZE = 4_096
SEED = 20260727

# Parameters obtained by solving E[ELU(X)] = 0 and Var[ELU(X)] = 1.
MU = -0.486326551925
SIGMA = 1.540156933960


def elu_in_place(values: np.ndarray) -> np.ndarray:
    """Apply the standard (alpha=1) ELU without allocating another full array."""
    negative = values <= 0
    np.expm1(values, out=values, where=negative)
    return values


def sample_elu_dot_products(rng: np.random.Generator) -> np.ndarray:
    """Draw independent vector pairs in batches and return their ELU dot products."""
    dots = np.empty(N_SAMPLES, dtype=np.float64)
    for start in range(0, N_SAMPLES, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_SAMPLES)
        shape = (stop - start, DIMENSION)
        x1 = rng.standard_normal(shape, dtype=np.float32)
        x2 = rng.standard_normal(shape, dtype=np.float32)
        x1 *= SIGMA
        x1 += MU
        x2 *= SIGMA
        x2 += MU
        elu_in_place(x1)
        elu_in_place(x2)
        dots[start:stop] = np.einsum("ij,ij->i", x1, x2, dtype=np.float64)
    return dots


def gaussian_dot_product_logpdf(z: np.ndarray, dimension: int) -> np.ndarray:
    r"""Exact log-PDF for Z = A dot B, with A,B iid N(0,I_dimension).

    The characteristic function is (1+t^2)^(-dimension/2), so Z has a
    symmetric variance-gamma distribution.  For even dimensions the Bessel-K
    expression can be evaluated stably using its half-integer finite sum.
    """
    if dimension % 2:
        raise ValueError("This stable finite-sum implementation needs even dimension")

    absolute_z = np.abs(np.asarray(z, dtype=np.float64))
    order = (dimension - 1) / 2
    m = dimension // 2 - 1
    result = np.empty_like(absolute_z)

    at_zero = absolute_z == 0
    result[at_zero] = (
        gammaln(order) - np.log(2) - 0.5 * np.log(np.pi) - gammaln(dimension / 2)
    )

    positive_z = absolute_z[~at_zero]
    k = np.arange(m + 1)[:, None]
    log_sum_terms = (
        gammaln(m + k + 1)
        - gammaln(k + 1)
        - gammaln(m - k + 1)
        - k * np.log(2 * positive_z[None, :])
    )
    log_bessel_k = (
        0.5 * (np.log(np.pi) - np.log(2) - np.log(positive_z))
        - positive_z
        + logsumexp(log_sum_terms, axis=0)
    )
    result[~at_zero] = (
        order * np.log(positive_z)
        + log_bessel_k
        - order * np.log(2)
        - 0.5 * np.log(np.pi)
        - gammaln(dimension / 2)
    )
    return result


rng = np.random.default_rng(SEED)
elu_dots = sample_elu_dot_products(rng)

# Use a fixed range in units of the common theoretical standard deviation sqrt(d).
reference_std = np.sqrt(DIMENSION)
plot_limit = 5 * reference_std
bins = np.linspace(-plot_limit, plot_limit, 161)
z = np.linspace(-plot_limit, plot_limit, 1_601)
reference_pdf = np.exp(gaussian_dot_product_logpdf(z, DIMENSION))

empirical_mean = elu_dots.mean()
empirical_std = elu_dots.std()
centered = elu_dots - empirical_mean
empirical_skew = np.mean(centered**3) / empirical_std**3

fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
ax.hist(
    elu_dots,
    bins=bins,
    density=True,
    alpha=0.48,
    color="#d35f2d",
    edgecolor="white",
    linewidth=0.35,
    label=rf"Monte Carlo: $\mathrm{{ELU}}(X_1)^\mathsf{{T}}\mathrm{{ELU}}(X_2)$ ({N_SAMPLES:,} pairs)",
)
ax.plot(
    z,
    reference_pdf,
    color="#315a9a",
    linewidth=2.4,
    label=rf"Exact PDF: $G_1^\mathsf{{T}}G_2$, $G_i\sim\mathcal{{N}}(0,I_{{{DIMENSION}}})$",
)
ax.axvline(0, color="0.3", linestyle="--", linewidth=0.9)
ax.set(
    title=f"Independent {DIMENSION}-D vector dot-product distributions",
    xlabel="dot product",
    ylabel="probability density",
    xlim=(-plot_limit, plot_limit),
)
ax.text(
    0.985,
    0.68,
    (
        "ELU dot product (empirical)\n"
        f"mean = {empirical_mean:.3f}\n"
        f"std = {empirical_std:.3f}\n"
        f"skewness = {empirical_skew:.3f}\n\n"
        "Gaussian reference (exact)\n"
        f"mean = 0\nstd = √1024 = {reference_std:.0f}"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
)
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=False)

output = Path(__file__).with_name("elu_dot_product_distribution.png")
fig.savefig(output, dpi=180)
print(f"samples={N_SAMPLES}")
print(f"ELU dot-product mean={empirical_mean:.8f}")
print(f"ELU dot-product std={empirical_std:.8f}")
print(f"ELU dot-product skewness={empirical_skew:.8f}")
print(f"Gaussian reference std={reference_std:.8f}")
print(output)
