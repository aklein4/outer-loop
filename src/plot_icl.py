import argparse
import json
import math
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

from utils import constants


BASE_PATH = os.path.join(constants.LOCAL_DATA_PATH, "icl_results")

# RUNS = {
#     "LoRA (1e-4)": "fresh/oloop-lora-llama3p2-1b-pre/base_lr_1e-04.json",
#     "Forte 100": "aklein4--Horizon-TPU_forte-v2-1b/000000000100.json",
#     "Forte 200": "aklein4--Horizon-TPU_forte-v2-1b/000000000200.json",
#     # "OLoop 500": "aklein4--Horizon-TPU_alpha/000000000500.json",
#     # "OLoop 500": "aklein4--Horizon-TPU_alpha/000000000500.json",
#     # "LoRA (1e-5)": "fresh/oloop-lora-llama3p2-1b-pre/base_lr_1e-05.json",
#     # "LoRA (3e-5)": "fresh/oloop-lora-llama3p2-1b-pre/base_lr_3e-05.json",
#     # "LoRA (3e-4)": "fresh/oloop-lora-llama3p2-1b-pre/base_lr_3e-04.json",
#     # "LoRA (1e-3)": "fresh/oloop-lora-llama3p2-1b-pre/base_lr_1e-03.json",
# }

COLOR_MAP = plt.get_cmap("viridis_r")
COLORBLIND_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
NUM_GRADIENT_COLORS = 9
_grad_index = 0
def gradient():
    global _grad_index
    color = COLOR_MAP(min(0.1+0.9*(_grad_index / NUM_GRADIENT_COLORS), 1.0))
    _grad_index += 1
    return color

RUNS = {
    "fresh/oloop-lora-llama3p2-1b-pre/base_lr_1e-04.json": {
        "label": "LoRA lr=1e-4", "color": "black"
    },

    "aklein4--Horizon-TPU_forte-v2-1b/000000000050.json": {
        "label": "Learned (new) step=050", "color": gradient()
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000100.json": {
        "label": "Learned (new) step=100", "color": gradient()
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000150.json": {
        "label": "Learned (new) step=150", "color": gradient()
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000200.json": {
        "label": "Learned (new) step=200", "color": gradient()
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000250.json": {
        "label": "Learned (new) step=250", "color": gradient()
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000400.json": {
        "label": "Learned (new) step=400", "color": gradient()
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000450.json": {
        "label": "Learned (new) step=450", "color": gradient()
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000500.json": {
        "label": "Learned (new) step=500", "color": gradient()
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000550.json": {
        "label": "Learned (new) step=550", "color": gradient()
    },
}

    # "aklein4--Horizon-TPU_forte-v2-fast-1b/000000000200.json": {
    #     "label": "Learned (fast) step=200", "color": "red"
    # },
    # "aklein4--Horizon-TPU_forte-v2-fast-1b/000000000400.json": {
    #     "label": "Learned (fast) step=400", "color": "red"
    # },
    # "aklein4--Horizon-TPU_forte-v2-fast-1b/000000000600.json": {
    #     "label": "Learned (fast) step=600", "color": "red"
    # },

    # "aklein4--Horizon-TPU_alpha/000000000250.json": {
    #     "label": "Learned (old) step=250", "color": "red"
    # },
    # "aklein4--Horizon-TPU_alpha/000000000500.json": {
    #     "label": "Learned (old) step=500", "color": "red"
    # },

# RUNS = {
#     # "fresh_pretrained_adam/oloop-lora-llama3p2-1b-pre/base_lr_1e-05.json": {
#     #     "label": "LoRA lr=1e-5", "color": gradient()
#     # },
#     # "fresh_pretrained_adam/oloop-lora-llama3p2-1b-pre/base_lr_3e-05.json": {
#     #     "label": "LoRA lr=3e-5", "color": gradient()
#     # },
#     # "fresh_pretrained_adam/oloop-lora-llama3p2-1b-pre/base_lr_1e-04.json": {
#     #     "label": "LoRA lr=1e-4", "color": gradient()
#     # },
#     "fresh_pretrained_adam/oloop-lora-llama3p2-1b-pre/base_lr_3e-04.json": {
#         "label": "LoRA lr=3e-4", "color": "black"
#     },
#     # "fresh_pretrained_adam/oloop-lora-llama3p2-1b-pre/base_lr_1e-03.json": {
#     #     "label": "LoRA lr=1e-3", "color": gradient()
#     # },
#     "aklein4--Horizon-TPU_forte-v2-freeze-1b/000000000050.json": {
#         "label": "Learned (freeze) step=50", "color": gradient()
#     },
#     "aklein4--Horizon-TPU_forte-v2-freeze-1b/000000000100.json": {
#         "label": "Learned (freeze) step=100", "color": gradient()
#     },
#     "aklein4--Horizon-TPU_forte-v2-freeze-1b/000000000150.json": {
#         "label": "Learned (freeze) step=150", "color": gradient()
#     },
#     "aklein4--Horizon-TPU_forte-v2-freeze-1b/000000000200.json": {
#         "label": "Learned (freeze) step=200", "color": gradient()
#     },
#     "aklein4--Horizon-TPU_forte-v2-freeze-1b/000000000250.json": {
#         "label": "Learned (freeze) step=250", "color": gradient()
#     },
# }

# RUNS = {
#     "fresh/oloop-lora-llama3p2-1b-pre/base_lr_1e-04.json": {
#         "label": "LoRA lr=1e-4", "color": "black"
#     },
    
#     "aklein4--Horizon-TPU_forte-v3-1b/000000000100.json": {
#         "label": "Learned (v3) step=100", "color": gradient()
#     },
#     "aklein4--Horizon-TPU_forte-v3-1b/000000000200.json": {
#         "label": "Learned (v3) step=200", "color": gradient()
#     },
#     "aklein4--Horizon-TPU_forte-v3-1b/000000000300.json": {
#         "label": "Learned (v3) step=300", "color": gradient()
#     },
# }

LORA_LABEL = "LoRA lr=1e-4"
LORA_REFERENCE_EXAMPLES = (16, 64, 1024)
CHECKPOINT_LABEL_RE = re.compile(r"^(?P<name>.+) step=(?P<step>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metric", default="average")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--y-axis", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def load_scores(path: Path, metric: str) -> tuple[list[int], list[float]]:
    
    with open(os.path.join(BASE_PATH, path), "r") as f:
        rows = json.load(f)
    
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list of evaluation rows")

    points = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a non-object row: {row!r}")
        if "num_examples" not in row:
            raise ValueError(f"{path} contains a row without num_examples")
        if metric not in row:
            raise ValueError(f"{path} contains a row without metric {metric!r}")
        points.append((int(row["num_examples"]), float(row[metric])))

    points.sort(key=lambda x: x[0])
    return [x for x, _ in points], [y for _, y in points]


def interpolate_y_at_x(x: list[int], y: list[float], target_x: float) -> float:
    """Interpolate y linearly with respect to log(x + 1)."""
    if not x or target_x < x[0] or target_x > x[-1]:
        raise ValueError(f"x={target_x:g} is outside [{x[0]}, {x[-1]}]")

    target_log_x = math.log1p(target_x)
    for left in range(len(x) - 1):
        if x[left] <= target_x <= x[left + 1]:
            if x[left] == target_x:
                return y[left]
            if x[left + 1] == target_x:
                return y[left + 1]
            left_log_x = math.log1p(x[left])
            fraction = (target_log_x - left_log_x) / (
                math.log1p(x[left + 1]) - left_log_x
            )
            return y[left] + fraction * (y[left + 1] - y[left])

    # The range check above means only a single-point series can reach here.
    if len(x) == 1 and x[0] == target_x:
        return y[0]
    raise ValueError(f"Could not interpolate x={target_x:g}")


def interpolate_x_at_y(x: list[int], y: list[float], target_y: float) -> float:
    """Find x for target_y, interpolating linearly in log(x + 1)."""
    for left in range(len(x) - 1):
        left_y, right_y = y[left], y[left + 1]
        if not min(left_y, right_y) <= target_y <= max(left_y, right_y):
            continue
        if left_y == target_y:
            return float(x[left])
        if right_y == target_y:
            return float(x[left + 1])
        if left_y == right_y:
            return float(x[left])

        fraction = (target_y - left_y) / (right_y - left_y)
        interpolated_log_x = math.log1p(x[left]) + fraction * (
            math.log1p(x[left + 1]) - math.log1p(x[left])
        )
        return math.expm1(interpolated_log_x)

    if len(x) == 1 and y[0] == target_y:
        return float(x[0])
    return float("nan")  # target_y is outside the range of y
    raise ValueError(f"Target loss {target_y:.6g} is outside the interpolation range")


def main(args):

    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10), constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.08, hspace=0.08)
    axes = axes.flatten()
    axes[1].sharey(axes[0])

    axes[0].axvline(65, color="black", linestyle="--")
    axes[1].axvline(64, color="black", linestyle="--")

    lines = {}
    for path, plot_kwargs in RUNS.items():
        x, y = load_scores(path, args.metric)

        if args.max_steps is not None:
            filtered = [(x_i, y_i) for x_i, y_i in zip(x, y) if x_i <= args.max_steps]
            x, y = zip(*filtered) if filtered else ([], [])
        if len(x) == 0:
            raise ValueError(f"No points found for {path} with max_steps={args.max_steps}")

        x, y = list(x), list(y)
        lines[plot_kwargs["label"]] = (x, y, plot_kwargs)

        axes[0].plot(
            [x_i + 1 for x_i in x], y, marker=".", markersize=10, **plot_kwargs
        )
        axes[1].plot(
            x, y, marker=".", markersize=10, **plot_kwargs
        )

    axes[0].set_xscale("log")
    axes[0].set_title("Log scale")

    axes[1].set_title("Linear scale")
    axes[1].legend()

    if LORA_LABEL not in lines:
        raise ValueError(f"RUNS must contain the LoRA reference label {LORA_LABEL!r}")
    lora_x, lora_y, _ = lines[LORA_LABEL]
    target_losses = {
        reference_examples: interpolate_y_at_x(
            lora_x, lora_y, reference_examples
        )
        for reference_examples in LORA_REFERENCE_EXAMPLES
    }

    checkpoint_summaries = {}
    for label, (x, y, plot_kwargs) in lines.items():
        match = CHECKPOINT_LABEL_RE.fullmatch(label)
        if match is None:
            continue
        efficiencies = {}
        for reference_examples, target_loss in target_losses.items():
            matched_examples = interpolate_x_at_y(x, y, target_loss)
            if matched_examples <= 0:
                raise ValueError(
                    f"{label!r} reaches the LoRA loss at {reference_examples} examples "
                    f"after {matched_examples:g} examples; relative sample efficiency "
                    "is undefined"
                )
            efficiencies[reference_examples] = reference_examples / matched_examples
        name = match.group("name")
        checkpoint_summaries.setdefault(name, []).append(
            (
                int(match.group("step")),
                interpolate_y_at_x(x, y, 0),
                efficiencies,
            )
        )

    for run_index, (name, points) in enumerate(checkpoint_summaries.items()):
        points.sort(key=lambda point: point[0])
        color = COLORBLIND_COLORS[run_index % len(COLORBLIND_COLORS)]
        axes[2].plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker=".",
            markersize=10,
            label=name,
            color=color,
        )
        for axis, reference_examples in zip(axes[3:], LORA_REFERENCE_EXAMPLES):
            axis.plot(
                [point[0] for point in points],
                [point[2][reference_examples] for point in points],
                marker=".",
                markersize=10,
                label=name,
                color=color,
            )

    axes[2].set_xscale("log")
    axes[2].set_title("Zero-shot performance")
    axes[2].set_xlabel("Meta-training steps")
    axes[2].set_ylabel(args.y_axis or "Loss (cross-entropy)")
    axes[2].grid(True, which="both", alpha=0.3)
    if checkpoint_summaries:
        axes[2].legend()

    for axis, reference_examples in zip(axes[3:], LORA_REFERENCE_EXAMPLES):
        axis.axhline(1, color="black", linestyle="--")
        axis.set_xscale("log")
        axis.set_title(
            f"Relative sample efficiency\n(versus LoRA @ {reference_examples})"
        )
        axis.set_xlabel("Meta-training steps")
        axis.set_ylabel(
            f"{reference_examples} / # examples to reach loss of LoRA "
            f"@ {reference_examples}"
        )
        axis.grid(True, which="both", alpha=0.3)
        if checkpoint_summaries:
            axis.legend()

    for ax in axes[:2]:
        ax.set_xlabel("Task examples seen")
        ax.grid(True, which="both", alpha=0.3)

    if args.y_axis is not None:
        axes[0].set_ylabel(args.y_axis)
    else:
        axes[0].set_ylabel("Loss (cross-entropy)")

    if args.title is not None:
        fig.suptitle(args.title)
    else:
        fig.suptitle("Supervised Learning Performance on Bitext Finetuning Datasets")

    plt.savefig("icl_plot.png", dpi=args.dpi)


if __name__ == "__main__":
    main(parse_args())
