import argparse
import json
from pathlib import Path
import os
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

RUNS = {
    "fresh/oloop-lora-llama3p2-1b-pre/base_lr_1e-04.json": {
        "label": "LoRA lr=1e-4", "color": "black"
    },
    # "aklein4--Horizon-TPU_alpha/000000000500.json": {
    #     "label": "Learned (old) step=500", "color": "red"
    # },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000050.json": {
        "label": "Learned (new) step=050", "color": COLOR_MAP(0.25)
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000100.json": {
        "label": "Learned (new) step=100", "color": COLOR_MAP(0.50)
    },
    "aklein4--Horizon-TPU_forte-v2-1b/000000000200.json": {
        "label": "Learned (new) step=200", "color": COLOR_MAP(0.75)
    },
}


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


def main(args):

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, constrained_layout=True,
    )

    axes[0].axvline(65, color="black", linestyle="--")
    axes[1].axvline(64, color="black", linestyle="--")

    for path, plot_kwargs in RUNS.items():
        x, y = load_scores(path, args.metric)

        if args.max_steps is not None:
            x, y = zip(*[(x_i, y_i) for x_i, y_i in zip(x, y) if x_i <= args.max_steps])
        if len(x) == 0:
            raise ValueError(f"No points found for {path} with max_steps={args.max_steps}")

        axes[0].plot(
            [x+1 for x in x], y, marker=".", markersize=10, **plot_kwargs
        )
        axes[1].plot(
            x, y, marker=".", markersize=10, **plot_kwargs
        )

    axes[0].set_xscale("log")
    axes[0].set_title("Log scale")

    axes[1].set_title("Linear scale")
    axes[1].legend()
    
    for ax in axes:
        ax.set_xlabel("Task Examples Seen")
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
