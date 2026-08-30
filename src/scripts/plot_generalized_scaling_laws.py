#!/usr/bin/env python3
"""Fit the generalized ICL scaling law and plot fits and sample efficiency."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src/local_data/icl_results"
DEFAULT_META_RUN = RESULTS_DIR / "aklein4--horizon-v2_piano-scaled"
DEFAULT_LORA_RUN = RESULTS_DIR / "fresh/oloop-lora-llama3p2-1b-pre"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures/scaling_laws"
REFERENCE_STEP = 50.0
MAX_FIT_STEP = 2300
MAX_EXAMPLES = 64
EXAMPLE_LEVELS = (0, 1, 2, 4, 8, 16, 32, 64)
EFFICIENCY_LEVELS = (8, 16, 32, 64)

RUN_STYLES = {
    "meta-learning": {"color": "tab:blue", "marker": "o"},
    "LoRA": {"color": "tab:red", "marker": "o"},
}

# Parameters are ordered as documented in generalized_meta_learning_scaling_law.md.
# These are optimization starts, not fixed values.
INITIAL_HINTS = {
    "meta-learning": {
        "L0_inf": 1.714,
        "A0": 0.083,
        "alpha0": 0.92,
        "L_inf_fraction": 1e-8,
        "beta": 0.06,
        "n0": 1.64,
        "gamma": 0.67,
        "delta": 0.26,
    },
    "LoRA": {
        "L0_inf": 1e-7,
        "A0": 1.772,
        "alpha0": 0.001,
        "L_inf_fraction": 0.5,
        "beta": -0.01,
        "n0": 1.87,
        "gamma": 1.17,
        "delta": 0.14,
    },
}

FULL_INITIAL_HINTS = {
    "meta-learning": {
        "L0_inf": 1.714,
        "L_inf_fraction": 1e-8,
        "A0": 0.083,
        "alpha0": 0.92,
        "M_offset": 1e-5,
        "g_inf": 2.0,
        "alpha_g": 0.06,
        "M_g_offset": 1e-5,
        "n0": 1.64,
        "gamma": 0.67,
        "delta": 0.26,
    },
    "LoRA": {
        "L0_inf": 1e-8,
        "L_inf_fraction": 1e-5,
        "A0": 1.772,
        "alpha0": 0.001,
        "M_offset": 1e-5,
        "g_inf": 0.5,
        "alpha_g": 0.019,
        "M_g_offset": 1e-5,
        "n0": 1.87,
        "gamma": 1.17,
        "delta": 0.14,
    },
}


def load_run(run_dir: Path) -> np.ndarray:
    """Return rows with columns (training_step, num_examples, average_loss)."""
    rows = []
    for path in sorted(run_dir.glob("*.json")):
        if not path.stem.isdigit():
            continue
        step = int(path.stem)
        if step > MAX_FIT_STEP:
            continue
        with path.open() as file:
            results = json.load(file)
        for result in results:
            num_examples = int(result["num_examples"])
            if num_examples <= MAX_EXAMPLES:
                rows.append((step, num_examples, float(result["average"])))
    if not rows:
        raise ValueError(f"No usable results found in {run_dir}")
    return np.asarray(rows, dtype=float)


def _logistic(value: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-np.logaddexp(0.0, -value))


def _positive(value: np.ndarray | float) -> np.ndarray | float:
    """Exponentiate an unconstrained variable with only machine-range clipping."""
    return np.exp(np.clip(value, -745.0, 700.0))


def unpack_parameters(raw: np.ndarray) -> dict[str, float]:
    l0_inf = float(_positive(raw[0]))
    return {
        "L0_inf": l0_inf,
        "A0": float(_positive(raw[2])),
        "alpha0": float(_positive(raw[3])),
        "L_inf": float(l0_inf * _logistic(raw[1])),
        "beta": float(raw[4]),
        "n0": float(_positive(raw[5])),
        "gamma": float(_positive(raw[6])),
        "delta": float(_positive(raw[7])),
    }


def pack_parameters(parameters: dict[str, float]) -> np.ndarray:
    fraction = min(max(parameters["L_inf_fraction"], 1e-300), 1 - 1e-12)
    logit = math.log(fraction) - math.log1p(-fraction)
    return np.asarray(
        [
            math.log(parameters["L0_inf"]),
            logit,
            math.log(parameters["A0"]),
            math.log(parameters["alpha0"]),
            parameters["beta"],
            math.log(parameters["n0"]),
            math.log(parameters["gamma"]),
            math.log(parameters["delta"]),
        ]
    )


def unpack_full_parameters(raw: np.ndarray) -> dict[str, float]:
    l0_inf = float(_positive(raw[0]))
    return {
        "L0_inf": l0_inf,
        "L_inf": float(l0_inf * _logistic(raw[1])),
        "A0": float(_positive(raw[2])),
        "alpha0": float(_positive(raw[3])),
        "M_offset": float(_positive(raw[4])),
        "g_inf": float(_positive(raw[5])),
        "alpha_g": float(_positive(raw[6])),
        "M_g_offset": float(_positive(raw[7])),
        "n0": float(_positive(raw[8])),
        "gamma": float(_positive(raw[9])),
        "delta": float(_positive(raw[10])),
    }


def pack_full_parameters(parameters: dict[str, float]) -> np.ndarray:
    fraction = min(max(parameters["L_inf_fraction"], 1e-300), 1 - 1e-12)
    logit = math.log(fraction) - math.log1p(-fraction)
    return np.asarray(
        [
            math.log(parameters["L0_inf"]),
            logit,
            math.log(parameters["A0"]),
            math.log(parameters["alpha0"]),
            math.log(parameters["M_offset"]),
            math.log(parameters["g_inf"]),
            math.log(parameters["alpha_g"]),
            math.log(parameters["M_g_offset"]),
            math.log(parameters["n0"]),
            math.log(parameters["gamma"]),
            math.log(parameters["delta"]),
        ]
    )


def predict(parameters: dict[str, float], steps, num_examples):
    steps = np.asarray(steps, dtype=float)
    num_examples = np.asarray(num_examples, dtype=float)
    if "M_offset" in parameters:
        step_ratio = (steps + parameters["M_offset"]) / (
            REFERENCE_STEP + parameters["M_offset"]
        )
    else:
        step_ratio = steps / REFERENCE_STEP
    initial_loss = parameters["L0_inf"] + parameters["A0"] * np.exp(
        -parameters["alpha0"] * np.log(step_ratio)
    )
    if "g_inf" in parameters:
        adaptation_ratio = (steps + parameters["M_g_offset"]) / (
            REFERENCE_STEP + parameters["M_g_offset"]
        )
        adaptation = parameters["g_inf"] + (1.0 - parameters["g_inf"]) * np.exp(
            -parameters["alpha_g"] * np.log(adaptation_ratio)
        )
    else:
        adaptation = np.exp(parameters["beta"] * np.log(step_ratio))
    effective_examples = adaptation * num_examples
    log_ratio = np.where(
        num_examples > 0,
        np.log(np.maximum(effective_examples, 1e-300)) - math.log(parameters["n0"]),
        -np.inf,
    )
    log_residual_fraction = -parameters["delta"] * np.logaddexp(
        0.0, parameters["gamma"] * log_ratio
    )
    residual_fraction = np.where(
        num_examples == 0, 1.0, np.exp(np.maximum(log_residual_fraction, -745.0))
    )
    return parameters["L_inf"] + (
        initial_loss - parameters["L_inf"]
    ) * residual_fraction


def fit_model(
    data: np.ndarray, label: str, starts: int = 64, full_model: bool = False
) -> dict[str, float]:
    target = data[:, 2]
    if full_model:
        base = pack_full_parameters(FULL_INITIAL_HINTS[label])
        unpack = unpack_full_parameters
        perturbation_scale = np.asarray(
            [1.5, 3.0, 2.0, 2.0, 5.0, 2.0, 2.0, 5.0, 2.0, 2.0, 2.0]
        )
    else:
        base = pack_parameters(INITIAL_HINTS[label])
        unpack = unpack_parameters
        perturbation_scale = np.asarray(
            [1.2, 3.0, 1.6, 1.8, 0.35, 1.8, 1.5, 1.5]
        )
    rng = np.random.default_rng(1729 if label == "meta-learning" else 2718)
    start_points = [base]
    if full_model:
        for model_offset in (1e-6, 10.0, 100.0, 1_000.0, 100_000.0):
            for adaptation_offset in (1e-6, 10.0, 100.0, 1_000.0):
                targeted = base.copy()
                targeted[4] = math.log(model_offset)
                targeted[7] = math.log(adaptation_offset)
                start_points.append(targeted)
    start_points.extend(
        base + rng.normal(size=base.size) * perturbation_scale
        for _ in range(max(starts - len(start_points), 0))
    )

    best = None
    for initial in start_points:
        try:
            result = least_squares(
                lambda raw: predict(
                    unpack(raw), data[:, 0], data[:, 1]
                )
                - target,
                initial,
                max_nfev=12_000,
                ftol=1e-13,
                xtol=1e-13,
                gtol=1e-13,
            )
        except (FloatingPointError, OverflowError, ValueError):
            continue
        residuals = predict(unpack(result.x), data[:, 0], data[:, 1]) - target
        sse = float(np.sum(np.square(residuals)))
        if np.isfinite(sse) and (best is None or sse < best[0]):
            best = (sse, result.x)
    if best is None:
        raise RuntimeError(f"Every optimization start failed for {label}")
    return unpack(best[1])


def model_metrics(parameters: dict[str, float], data: np.ndarray) -> dict[str, float]:
    residuals = predict(parameters, data[:, 0], data[:, 1]) - data[:, 2]
    sse = float(np.sum(np.square(residuals)))
    count = len(data)
    return {
        "observations": count,
        "rmse": float(np.sqrt(sse / count)),
        "mae": float(np.mean(np.abs(residuals))),
        "bias": float(np.mean(residuals)),
        "bic": float(
            count * math.log(sse / count) + len(parameters) * math.log(count)
        ),
    }


def observed_curves(data: np.ndarray) -> dict[int, list[tuple[int, float]]]:
    curves = {}
    for step, num_examples, loss in data:
        curves.setdefault(int(num_examples), []).append((int(step), float(loss)))
    for points in curves.values():
        points.sort()
    return curves


def plot_loss_fits(
    datasets: dict[str, np.ndarray],
    fits: dict[str, dict[str, float]],
    output: Path,
    extrapolation_factor: int,
) -> None:
    columns = 4
    rows = math.ceil(len(EXAMPLE_LEVELS) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 3.8 * rows),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    max_step = MAX_FIT_STEP * extrapolation_factor
    fit_steps = np.geomspace(REFERENCE_STEP, max_step, 500)
    curves = {label: observed_curves(data) for label, data in datasets.items()}

    for index, (axis, num_examples) in enumerate(zip(axes.flat, EXAMPLE_LEVELS)):
        for label in ("meta-learning", "LoRA"):
            style = RUN_STYLES[label]
            points = curves[label][num_examples]
            axis.scatter(
                [step for step, _ in points],
                [loss for _, loss in points],
                s=25,
                color=style["color"],
                alpha=0.8,
                zorder=3,
                label=label if index == 0 else None,
            )
            axis.plot(
                fit_steps,
                predict(fits[label], fit_steps, np.full_like(fit_steps, num_examples)),
                linestyle="--",
                linewidth=1.8,
                color=style["color"],
                zorder=2,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(REFERENCE_STEP / 1.12, max_step * 1.12)
        axis.set_title(f"num_examples = {num_examples}")
        axis.grid(True, which="both", alpha=0.3)
        if index // columns == rows - 1:
            axis.set_xlabel("Training step (log scale)")
        if index % columns == 0:
            axis.set_ylabel("Average loss")

    axes.flat[0].legend(fontsize="small")
    suffix = " with 100× checkpoint extrapolation" if extrapolation_factor > 1 else ""
    fig.suptitle(f"Task loss and generalized scaling-law fits{suffix}")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def examples_at_loss(examples: list[int], losses: list[float], target: float) -> float:
    """Interpolate the first observed crossing in log-log space."""
    for index in range(len(examples) - 1):
        left_loss, right_loss = losses[index : index + 2]
        if not min(left_loss, right_loss) <= target <= max(left_loss, right_loss):
            continue
        if target == left_loss or left_loss == right_loss:
            return float(examples[index])
        if target == right_loss:
            return float(examples[index + 1])
        fraction = (math.log(target) - math.log(left_loss)) / (
            math.log(right_loss) - math.log(left_loss)
        )
        return math.expm1(
            math.log1p(examples[index])
            + fraction
            * (math.log1p(examples[index + 1]) - math.log1p(examples[index]))
        )
    return float("nan")


def empirical_efficiencies(
    meta_data: np.ndarray, lora_data: np.ndarray
) -> dict[int, list[tuple[int, float]]]:
    efficiencies = {n: [] for n in EFFICIENCY_LEVELS}
    common_steps = sorted(set(meta_data[:, 0].astype(int)) & set(lora_data[:, 0].astype(int)))
    for step in common_steps:
        meta_step = meta_data[meta_data[:, 0] == step]
        lora_step = lora_data[lora_data[:, 0] == step]
        meta_points = sorted((int(n), float(loss)) for _, n, loss in meta_step)
        lora_loss = {int(n): float(loss) for _, n, loss in lora_step}
        examples = [n for n, _ in meta_points]
        losses = [loss for _, loss in meta_points]
        for reference_examples in EFFICIENCY_LEVELS:
            required = examples_at_loss(
                examples, losses, lora_loss[reference_examples]
            )
            efficiencies[reference_examples].append(
                (step, reference_examples / required)
            )
    return efficiencies


def fitted_efficiency(
    steps: np.ndarray,
    reference_examples: int,
    meta_fit: dict[str, float],
    lora_fit: dict[str, float],
) -> np.ndarray:
    target_loss = predict(
        lora_fit, steps, np.full_like(steps, reference_examples, dtype=float)
    )
    if "M_offset" in meta_fit:
        model_ratio = (steps + meta_fit["M_offset"]) / (
            REFERENCE_STEP + meta_fit["M_offset"]
        )
        adaptation_ratio = (steps + meta_fit["M_g_offset"]) / (
            REFERENCE_STEP + meta_fit["M_g_offset"]
        )
        meta_adaptation = meta_fit["g_inf"] + (1.0 - meta_fit["g_inf"]) * np.exp(
            -meta_fit["alpha_g"] * np.log(adaptation_ratio)
        )
    else:
        model_ratio = steps / REFERENCE_STEP
        meta_adaptation = np.exp(
            meta_fit["beta"] * np.log(steps / REFERENCE_STEP)
        )
    meta_l0 = meta_fit["L0_inf"] + meta_fit["A0"] * np.exp(
        -meta_fit["alpha0"] * np.log(model_ratio)
    )
    residual_fraction = (target_loss - meta_fit["L_inf"]) / (
        meta_l0 - meta_fit["L_inf"]
    )
    valid = (residual_fraction > 0) & (residual_fraction <= 1)
    required = np.full_like(steps, np.nan, dtype=float)
    required[valid] = (
        meta_fit["n0"]
        / meta_adaptation[valid]
        * np.power(
            np.power(residual_fraction[valid], -1.0 / meta_fit["delta"]) - 1.0,
            1.0 / meta_fit["gamma"],
        )
    )
    return reference_examples / required


def plot_efficiency(
    datasets: dict[str, np.ndarray],
    fits: dict[str, dict[str, float]],
    output: Path,
    extrapolation_factor: int,
) -> None:
    empirical = empirical_efficiencies(
        datasets["meta-learning"], datasets["LoRA"]
    )
    max_step = MAX_FIT_STEP * extrapolation_factor
    fit_steps = np.geomspace(REFERENCE_STEP, max_step, 500)
    fig, axes = plt.subplots(
        1,
        len(EFFICIENCY_LEVELS),
        figsize=(16, 3.8),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    for index, (axis, reference_examples) in enumerate(
        zip(axes.flat, EFFICIENCY_LEVELS)
    ):
        points = empirical[reference_examples]
        axis.scatter(
            [step for step, efficiency in points if np.isfinite(efficiency)],
            [efficiency for step, efficiency in points if np.isfinite(efficiency)],
            s=28,
            color="tab:blue",
            alpha=0.8,
            zorder=3,
            label="empirical" if index == 0 else None,
        )
        axis.plot(
            fit_steps,
            fitted_efficiency(
                fit_steps,
                reference_examples,
                fits["meta-learning"],
                fits["LoRA"],
            ),
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
            label="scaling-law fit" if index == 0 else None,
        )
        axis.axhline(1, color="0.35", linestyle=":", linewidth=1)
        axis.set_xscale("log")
        axis.set_xlim(REFERENCE_STEP / 1.12, max_step * 1.12)
        axis.set_title(f"× fewer examples than LoRA @ {reference_examples}")
        axis.set_xlabel("Training step (log scale)")
        if index == 0:
            axis.set_ylabel("Sample efficiency")
        axis.grid(True, which="both", alpha=0.3)

    axes.flat[0].legend(fontsize="small")
    suffix = " with 100× checkpoint extrapolation" if extrapolation_factor > 1 else ""
    fig.suptitle(f"Meta-learning sample efficiency relative to LoRA{suffix}")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_parameters(
    output: Path,
    run_dirs: dict[str, Path],
    datasets: dict[str, np.ndarray],
    fits: dict[str, dict[str, float]],
    full_model: bool,
) -> None:
    if full_model:
        l0_formula = (
            "L0(inf) + A0 * ((M + M_offset) / (M0 + M_offset))^-alpha0"
        )
        adaptation_formula = (
            "g_inf + (1 - g_inf) * "
            "((M + M_g_offset) / (M0 + M_g_offset))^-alpha_g"
        )
        extra_constraints = [
            "M_offset >= 0",
            "g_inf > 0",
            "alpha_g > 0",
            "M_g_offset >= 0",
        ]
    else:
        l0_formula = "L0(inf) + A0 * (M / M0)^-alpha0"
        adaptation_formula = "(M / M0)^beta"
        extra_constraints = []
    payload = {
        "model": {
            "loss": "L_inf + (L0(M) - L_inf) * [1 + (g(M) * n / n0)^gamma]^-delta",
            "L0": l0_formula,
            "g": adaptation_formula,
            "M0": REFERENCE_STEP,
            "constraints": [
                "A0 > 0",
                "alpha0 > 0",
                "n0 > 0",
                "gamma > 0",
                "delta > 0",
                "0 <= L_inf <= L0(inf)",
            ]
            + extra_constraints,
        },
        "fit_scope": {
            "max_training_step": MAX_FIT_STEP,
            "max_num_examples": MAX_EXAMPLES,
            "num_example_levels": list(EXAMPLE_LEVELS),
            "loss_weighting": "unweighted",
        },
        "fits": {},
    }
    for label in ("meta-learning", "LoRA"):
        parameters = dict(fits[label])
        parameters["L0(inf)"] = parameters.pop("L0_inf")
        payload["fits"][label] = {
            "source_run": str(run_dirs[label].relative_to(REPO_ROOT)),
            "parameters": parameters,
            "metrics": model_metrics(fits[label], datasets[label]),
        }
    with output.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-run", type=Path, default=DEFAULT_META_RUN)
    parser.add_argument("--lora-run", type=Path, default=DEFAULT_LORA_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Fit checkpoint offsets and the full offset-floor adaptation law.",
    )
    parser.add_argument(
        "--filename-suffix",
        default="",
        help="Suffix inserted before each output extension, for example _full.",
    )
    args = parser.parse_args()

    run_dirs = {"meta-learning": args.meta_run, "LoRA": args.lora_run}
    datasets = {label: load_run(path) for label, path in run_dirs.items()}
    fits = {
        label: fit_model(
            datasets[label], label, starts=args.starts, full_model=args.full_model
        )
        for label in ("meta-learning", "LoRA")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.filename_suffix
    plot_loss_fits(
        datasets,
        fits,
        args.output_dir / f"training_progress_fit{suffix}.png",
        extrapolation_factor=1,
    )
    plot_loss_fits(
        datasets,
        fits,
        args.output_dir / f"training_progress_fit_100x{suffix}.png",
        extrapolation_factor=100,
    )
    plot_efficiency(
        datasets,
        fits,
        args.output_dir / f"sample_efficiency_fit{suffix}.png",
        extrapolation_factor=1,
    )
    plot_efficiency(
        datasets,
        fits,
        args.output_dir / f"sample_efficiency_fit_100x{suffix}.png",
        extrapolation_factor=100,
    )
    write_parameters(
        args.output_dir / f"fit_parameters{suffix}.json",
        run_dirs,
        datasets,
        fits,
        args.full_model,
    )
    for label in ("meta-learning", "LoRA"):
        print(label, fits[label], model_metrics(fits[label], datasets[label]))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
