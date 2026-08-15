#!/usr/bin/env python3
"""Plot ICL loss and sample efficiency from a W&B run's episode losses."""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import wandb


plt.style.use("tableau-colorblind10")
EPISODE_RE = re.compile(r"^lm_loss/episode_(\d+)$")
REFERENCE_EXAMPLES = (2, 4, 8, 16, 32, 64)
DEFAULT_RUN = "/aklein4/horizon-v2/runs/06lwno1t"
FIT_RANGE_COUNT = 10


def episode_keys(run):
    columns = run.history(samples=1, pandas=True).columns
    numbered = sorted(
        (int(match.group(1)), key)
        for key in columns
        if (match := EPISODE_RE.fullmatch(key))
    )
    if not numbered:
        raise ValueError("The run has no lm_loss/episode_n history fields")
    return numbered


def load_history(run):
    numbered = episode_keys(run)
    keys = ["_step", *(key for _, key in numbered)]
    frame = pd.DataFrame(run.scan_history(keys=keys))
    frame[keys] = frame[keys].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["_step"]).sort_values("_step")
    if frame.empty:
        raise ValueError("The run has no usable history rows")
    return frame, numbered


def evenly_spaced_ranges(frame, count):
    steps = np.sort(frame["_step"].unique())
    if len(steps) < count:
        raise ValueError(f"Need at least {count} logged steps; found {len(steps)}")
    chunks = np.array_split(steps, count)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks]


def range_curves(frame, numbered, ranges):
    curves = []
    keys = [key for _, key in numbered]
    # episode_00 is the loss after the first task example, and episode_63 is
    # therefore the loss after 64 task examples.
    episodes = np.asarray([episode + 1 for episode, _ in numbered], dtype=float)
    for start, end in ranges:
        window = frame[frame["_step"].between(start, end)]
        losses = window[keys].mean(axis=0, skipna=True).to_numpy(dtype=float)
        curves.append({
            "start": start,
            "end": end,
            "step": (start + end) / 2 + 1,
            "episodes": episodes,
            "losses": losses,
        })
    return curves


def y_at_x(x, y, target):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if target < x.min() or target > x.max():
        raise ValueError(f"c'={target:g} is outside [{x.min():g}, {x.max():g}]")
    return float(np.interp(math.log1p(target), np.log1p(x), y))


def x_at_y(x, y, target):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    for index in range(len(x) - 1):
        left, right = y[index], y[index + 1]
        if min(left, right) <= target <= max(left, right):
            if left == right:
                return float(x[index])
            fraction = (target - left) / (right - left)
            return float(np.expm1(
                np.log1p(x[index])
                + fraction * (np.log1p(x[index + 1]) - np.log1p(x[index]))
            ))
    return float("nan")


def context_amplitude_at_t(t, fit):
    return np.full_like(np.asarray(t, dtype=float), fit["amplitude"], dtype=float)


def context_exponent_at_t(t, fit):
    relative_t = np.asarray(t, dtype=float) / fit["reference_step"]
    return (
        fit["beta_bias"]
        + fit["beta_scale"] * relative_t ** fit["beta_rate"]
    )


def fit_scaling_law(curves):
    observations = [
        (c, curve["step"], loss)
        for curve in curves
        for c, loss in zip(curve["episodes"], curve["losses"])
        if c > 0 and np.isfinite(loss)
    ]
    c, t, loss = (np.asarray(values) for values in zip(*observations))
    reference_step = curves[0]["step"]

    def residual(parameters):
        (gamma, log_s_magnitude, log_alpha,
         log_amplitude,
         log_beta_bias, log_beta_scale, beta_rate) = parameters
        alpha = np.exp(log_alpha)
        relative_t = t / reference_step
        amplitude = np.exp(log_amplitude)
        beta_t = (
            np.exp(log_beta_bias)
            + np.exp(log_beta_scale) * relative_t ** beta_rate
        )
        denominator = amplitude * c ** -beta_t + t ** -alpha
        return (
            gamma - np.exp(log_s_magnitude) / denominator - loss
        )

    bounds = (
        [-np.inf, -20, -12, -20, -12, -12, -10],
        [np.inf, 20, 3, 20, 3, 3, 10],
    )
    starts = (
        [2.45, -6.76, 0.55, -6.16, -1.74, -1.78, -2.81],
        [2.3, -1.0, 0.5, -0.5, -2.0, -2.0, -0.5],
    )
    candidates = [
        least_squares(
            residual,
            start,
            bounds=bounds,
            max_nfev=100_000,
        )
        for start in starts
    ]
    result = min(candidates, key=lambda candidate: np.sum(candidate.fun ** 2))
    if not result.success:
        raise RuntimeError(f"Scaling-law fit failed: {result.message}")
    (gamma, log_s_magnitude, log_alpha,
     log_amplitude,
     log_beta_bias, log_beta_scale, beta_rate) = result.x
    alpha = np.exp(log_alpha)
    return {
        "gamma": gamma,
        "s": -np.exp(log_s_magnitude),
        "alpha": alpha,
        "amplitude": np.exp(log_amplitude),
        "beta_bias": np.exp(log_beta_bias),
        "beta_scale": np.exp(log_beta_scale),
        "beta_rate": beta_rate,
        "beta_model": "biased-power beta(t) only",
        "reference_step": reference_step,
        "count": len(observations),
        "rmse": np.sqrt(np.mean(result.fun ** 2)),
    }


def fitted_efficiency(steps, reference, reference_step, fit):
    steps = np.atleast_1d(np.asarray(steps, dtype=float))
    amplitude = context_amplitude_at_t(steps, fit)
    reference_amplitude = context_amplitude_at_t(reference_step, fit)
    beta = context_exponent_at_t(steps, fit)
    reference_beta = context_exponent_at_t(reference_step, fit)
    target_context_term = (
        reference_amplitude * reference ** -reference_beta
        + reference_step ** -fit["alpha"]
        - steps ** -fit["alpha"]
    )
    equivalent_context = np.where(
        target_context_term > 0,
        (amplitude / target_context_term) ** (1 / beta),
        np.nan,
    )
    return np.where(
        np.isfinite(equivalent_context) & (equivalent_context > 0),
        reference / equivalent_context,
        np.nan,
    )


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def print_fit(fit, reference_step):
    g = scientific(fit["gamma"])
    s, alpha = map(scientific, (fit["s"], fit["alpha"]))
    tr = scientific(reference_step)
    amplitude = scientific(fit["amplitude"])
    beta_bias, beta_scale, beta_rate = map(scientific, (
        fit["beta_bias"], fit["beta_scale"], fit["beta_rate"]
    ))
    print(
        f"{fit['beta_model'].capitalize()} fit "
        f"({fit['count']} observations, RMSE={fit['rmse']:.6g}):"
    )
    print(f"  A = {amplitude}")
    print(
        f"  beta(t) = {beta_bias} + {beta_scale} "
        f"(t / {tr})^{beta_rate}"
    )
    print(f"  S = {s}")
    print(
        f"  L_t(c) = {g} + S / (A c^-beta(t) + t^-{alpha})"
    )


def make_figure(curves, fit, run_name):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    axes = axes.flatten()
    colors = plt.get_cmap("viridis_r")(np.linspace(0.1, 0.9, len(curves)))
    for curve, color in zip(curves, colors):
        label = f"steps {curve['start']}-{curve['end']}"
        axes[0].plot(curve["episodes"], curve["losses"], color=color, label=label)
        axes[1].plot(curve["episodes"], curve["losses"], color=color, label=label)
    axes[0].set(xscale="log", title="Log scale", xlabel="Task examples seen", ylabel="Loss")
    axes[1].set(title="Linear scale", xlabel="Task examples seen")
    axes[1].legend(fontsize="small", ncols=2)

    efficiency_curves = curves[-FIT_RANGE_COUNT:]
    reference_curve = efficiency_curves[0]
    axes[2].plot(
        [curve["step"] for curve in curves],
        [y_at_x(curve["episodes"], curve["losses"], 1) for curve in curves],
        ".-", markersize=10,
    )
    axes[2].set(xscale="log", title="First-example performance",
                xlabel="Meta-training step", ylabel="Loss")

    smooth_steps = np.geomspace(
        efficiency_curves[0]["step"], efficiency_curves[-1]["step"], 300
    )
    for axis, reference in zip(axes[3:], REFERENCE_EXAMPLES):
        target = y_at_x(reference_curve["episodes"], reference_curve["losses"], reference)
        efficiencies = [
            reference / x_at_y(curve["episodes"], curve["losses"], target)
            for curve in efficiency_curves
        ]
        axis.plot([curve["step"] for curve in efficiency_curves], efficiencies,
                  ".-", markersize=10, label="Measured")
        fitted = fitted_efficiency(
            smooth_steps, reference, reference_curve["step"], fit
        )
        axis.plot(smooth_steps, fitted, "k--", linewidth=2,
            label=f"Scaling-law fit ({fit['beta_model']})")
        axis.axhline(1, color="0.5", linestyle=":")
        axis.set(xscale="log",
                 title=f"Relative sample efficiency\n(reference @ {reference})",
                 xlabel="Meta-training step",
                 ylabel=f"{reference} / # examples to reach reference loss")
        axis.legend()
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"W&B run {run_name}: loss and task-example efficiency")
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", default=DEFAULT_RUN,
                        help="W&B run path, e.g. /entity/project/runs/id")
    parser.add_argument("--output", type=Path, default=Path("figures/wandb_icl_plot.png"))
    parser.add_argument("--num-ranges", type=int, default=11)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    run = wandb.Api().run(args.run)
    frame, numbered = load_history(run)
    ranges = evenly_spaced_ranges(frame, args.num_ranges)
    curves = range_curves(frame, numbered, ranges)
    if len(curves) < FIT_RANGE_COUNT:
        raise ValueError(f"Need at least {FIT_RANGE_COUNT} ranges for the fit")
    fit_curves = curves[-FIT_RANGE_COUNT:]
    fit = fit_scaling_law(fit_curves)
    print_fit(fit, fit_curves[0]["step"])
    figure = make_figure(curves, fit, "/".join(run.path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
