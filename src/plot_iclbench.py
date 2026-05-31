import os
import json
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from utils import constants


RESULTS_DIR = os.path.join(constants.LOCAL_DATA_PATH, "ManyICLBench_results_50")

SAVE_PATH = "iclbench_accuracy_by_context_50.png"

STYLE_MAP = {
    "OLoop": {"color": "black", "label": "OLoop (ours)"},
    "LoRA": {"color": "blue", "label": "LoRA"},
    "ICL": {"color": "red", "label": "ICL"},
    "Sliding": {"color": "green", "label": "Sliding"},
}

BASELINE_CONTEXT_LABEL = "128k"

LINEAR = False


def context_length_value(name):
    return int(name[:-1]) * 1024


def find_result_json(version_dir):
    result_files = sorted(version_dir.glob("*/*/ManyICLBench.json"))
    if not result_files:
        return None
    if len(result_files) > 1:
        print(f"Found multiple result files under {version_dir}; using {result_files[0]}")
    return result_files[0]


def load_results(results_dir):
    results_dir = Path(results_dir)

    lines = {}

    for context_dir in sorted(results_dir.iterdir(), key=lambda path: context_length_value(path.name)):
        if not context_dir.is_dir():
            continue

        context_length = context_length_value(context_dir.name)

        for version_dir in sorted(context_dir.iterdir()):
            if not version_dir.is_dir():
                continue

            result_file = find_result_json(version_dir)
            if result_file is None:
                continue

            with result_file.open() as f:
                result = json.load(f)

            lines.setdefault(version_dir.name, []).append(
                {
                    "context_label": context_dir.name,
                    "context_length": context_length,
                    "accuracy": result["accuracy"],
                    "seen": result.get("seen"),
                    "path": result_file,
                }
            )

    return {
        version: sorted(points, key=lambda point: point["context_length"])
        for version, points in sorted(lines.items())
    }


def interpolated_context_for_accuracy(points, target_accuracy):
    points = sorted(points, key=lambda point: point["context_length"])

    for left, right in zip(points, points[1:]):
        left_accuracy = left["accuracy"]
        right_accuracy = right["accuracy"]

        if left_accuracy == target_accuracy:
            return left["context_length"]

        crosses_target = (
            min(left_accuracy, right_accuracy)
            <= target_accuracy
            <= max(left_accuracy, right_accuracy)
        )
        if not crosses_target:
            continue

        if left_accuracy == right_accuracy:
            return left["context_length"]

        interpolation_fraction = (
            (target_accuracy - left_accuracy) / (right_accuracy - left_accuracy)
        )
        return left["context_length"] + interpolation_fraction * (
            right["context_length"] - left["context_length"]
        )

    if points[-1]["accuracy"] == target_accuracy:
        return points[-1]["context_length"]

    raise ValueError(
        f"Target accuracy {target_accuracy:.4f} is outside the interpolation range"
    )


def print_sample_efficiency(lines):
    baseline_context = context_length_value(BASELINE_CONTEXT_LABEL)
    baseline_accuracy = next(
        point["accuracy"]
        for point in lines["LoRA"]
        if point["context_length"] == baseline_context
    )
    oloop_context = interpolated_context_for_accuracy(
        lines["OLoop"],
        baseline_accuracy,
    )
    sample_efficiency = baseline_context / oloop_context

    print(
        f"\nOLoop is {sample_efficiency:.2f}x more sample-efficient than LoRA "
        f"at {BASELINE_CONTEXT_LABEL} "
        f"(LoRA accuracy {100 * baseline_accuracy:.1f}%, "
        f"matched by OLoop at {oloop_context / 1024:.1f}k)."
    )


def plot_results(lines, output_path):
    plt.figure(figsize=(7, 6))

    for version, points in lines.items():
        x = [point["context_length"] for point in points]
        y = [100 * point["accuracy"] for point in points]
        plt.plot(x, y, marker="s", **STYLE_MAP[version])

    all_x = sorted({point["context_length"] for points in lines.values() for point in points})
    labels_by_x = {
        point["context_length"]: point["context_label"]
        for points in lines.values()
        for point in points
    }

    if LINEAR:
        plt.xticks(
            np.linspace(10*1024, 130*1024, 7),
            [f"{int(x)//1024:d}k" for x in np.linspace(10*1024, 130*1024, 7)],
        )
    else:
        plt.xscale("log", base=2)
        plt.xticks(all_x, [labels_by_x[x] for x in all_x])

    plt.xlabel("Context Length")
    plt.ylabel("Accuracy (%)")
    plt.title(f"ManyICLBench Accuracy by Context Length ({'linear' if LINEAR else 'log'} scale)")
    
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)


def main():

    l = load_results(RESULTS_DIR)
    lines = {}
    for k in STYLE_MAP.keys():
        lines[k] = l[k]

    print("\nLoaded results:")
    context_labels = sorted(
        {point["context_label"] for points in lines.values() for point in points},
        key=context_length_value,
    )
    version_width = max(len("Version"), *(len(version) for version in lines))
    value_width = max(len(label) for label in context_labels)

    header = "  " + "Version".ljust(version_width)
    header += "  " + "  ".join(label.rjust(value_width) for label in context_labels)
    print(header)

    for version, points in lines.items():
        accuracies = {
            point["context_label"]: f"{100 * point['accuracy']:.1f}"
            for point in points
        }
        values = [
            accuracies.get(label, "-").rjust(value_width)
            for label in context_labels
        ]
        print(f"  {version.ljust(version_width)}  {'  '.join(values)}")

    try:
        print_sample_efficiency(lines)
    except:
        pass

    plot_results(lines, SAVE_PATH)
    print(f"\nPlot saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
