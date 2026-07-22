import argparse
from pathlib import Path

import datasets
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASETS = [
    "aklein4/few-shot-TrackStar-8192-norm",
    "aklein4/few-shot-TrackStar-8192-test",
]
DEFAULT_COLUMN = "similarity_to_cluster_mean"
DEFAULT_OUTPUT = Path("similarity_to_cluster_mean_cdf.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot empirical CDF curves for a numeric column across all dataset configs and splits."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Hugging Face dataset repos to compare.",
    )
    parser.add_argument("--column", default=DEFAULT_COLUMN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def display_name(dataset_name):
    return dataset_name.rsplit("/", 1)[-1]


def load_column_values(dataset_name, column):
    arrays = []
    sources = []

    configs = datasets.get_dataset_config_names(dataset_name)
    for config in configs:
        splits = datasets.get_dataset_split_names(dataset_name, config)
        for split in splits:
            ds = datasets.load_dataset(dataset_name, config, split=split)
            if column not in ds.column_names:
                raise KeyError(
                    f"{dataset_name} / {config} / {split} does not contain {column!r}"
                )

            values = np.asarray(ds[column], dtype=np.float64)
            values = values[np.isfinite(values)]
            arrays.append(values)
            sources.append(f"{config}:{split} ({len(values):,})")

    if not arrays:
        raise RuntimeError(f"No configs or splits found for {dataset_name}")

    return np.concatenate(arrays), sources


def plot_cdfs(values_by_label, sources_by_label, column, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    colors = ["#2f6f73", "#c94f4f", "#6b5fb5", "#d18f2f", "#4f7fbf"]

    for i, (label, values) in enumerate(values_by_label.items()):
        sorted_values = np.sort(values)
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.plot(
            sorted_values,
            cumulative,
            label=(
                f"{label} (n={len(values):,}, mean={values.mean():.4f}, "
                f"median={np.median(values):.4f})"
            ),
            color=colors[i % len(colors)],
            linewidth=2.0,
        )

    ax.set_title("Similarity to Cluster Mean Empirical CDF", fontsize=16, pad=14)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Cumulative fraction of examples", fontsize=12)
    ax.legend(frameon=True, loc="lower right")
    ax.set_ylim(0, 1)
    ax.margins(x=0.01)

    caption = "Included subsets/splits: " + " | ".join(
        f"{label}: {', '.join(sources)}"
        for label, sources in sources_by_label.items()
    )
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    values_by_label = {}
    sources_by_label = {}
    for dataset_name in args.datasets:
        label = display_name(dataset_name)
        values, sources = load_column_values(dataset_name, args.column)
        values_by_label[label] = values
        sources_by_label[label] = sources

    plot_cdfs(
        values_by_label=values_by_label,
        sources_by_label=sources_by_label,
        column=args.column,
        output=args.output,
    )

    print(args.output)
    for label, values in values_by_label.items():
        print(
            f"{label}: n={len(values):,} min={values.min():.6f} "
            f"max={values.max():.6f} mean={values.mean():.6f} std={values.std():.6f}"
        )


if __name__ == "__main__":
    main()
