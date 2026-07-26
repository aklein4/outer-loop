"""Plot layer-wise RMS values for the matrix parameters of a Llama model."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PARAMETER_GROUPS = {
    "Attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "MLP": ("gate_proj", "up_proj", "down_proj"),
}
TENSOR_NAME = re.compile(
    r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("llama_3_2_1b_instruct_parameter_rms.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output data path (defaults to the plot path with a .csv suffix)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download missing model files",
    )
    return parser.parse_args()


def resolve_model_dir(model: str, local_files_only: bool) -> Path:
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        return model_path
    return Path(
        snapshot_download(
            repo_id=model,
            allow_patterns=("config.json", "*.safetensors", "*.safetensors.index.json"),
            local_files_only=local_files_only,
        )
    )


def find_weight_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as file:
            weight_map = json.load(file)["weight_map"]
        return [model_dir / name for name in sorted(set(weight_map.values()))]

    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}")
    return weight_files


def compute_rms(weight_files: list[Path]) -> dict[str, dict[int, float]]:
    parameter_rms: dict[str, dict[int, float]] = {
        parameter: {}
        for parameters in PARAMETER_GROUPS.values()
        for parameter in parameters
    }
    with torch.no_grad():
        for weight_file in weight_files:
            with safe_open(weight_file, framework="pt", device="cpu") as tensors:
                for tensor_name in tensors.keys():
                    match = TENSOR_NAME.match(tensor_name)
                    if match is None:
                        continue
                    layer, parameter = match.groups()
                    weight = tensors.get_tensor(tensor_name).float()
                    parameter_rms[parameter][int(layer)] = weight.square().mean().sqrt().item()
    return parameter_rms


def validate(parameter_rms: dict[str, dict[int, float]], num_layers: int) -> None:
    expected_layers = set(range(num_layers))
    errors = []
    for parameter, values in parameter_rms.items():
        missing = sorted(expected_layers - values.keys())
        extra = sorted(values.keys() - expected_layers)
        if missing or extra:
            errors.append(f"{parameter}: missing={missing}, extra={extra}")
    if errors:
        raise ValueError("Unexpected layer coverage:\n" + "\n".join(errors))


def write_csv(path: Path, parameter_rms: dict[str, dict[int, float]], num_layers: int) -> None:
    parameters = [parameter for group in PARAMETER_GROUPS.values() for parameter in group]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("layer", *parameters))
        for layer in range(num_layers):
            writer.writerow((layer, *(f"{parameter_rms[p][layer]:.10g}" for p in parameters)))


def make_plot(
    path: Path,
    model_name: str,
    parameter_rms: dict[str, dict[int, float]],
    num_layers: int,
) -> None:
    layers = list(range(num_layers))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    colors = {
        "q_proj": "#0072B2",
        "k_proj": "#E69F00",
        "v_proj": "#009E73",
        "o_proj": "#CC79A7",
        "gate_proj": "#D55E00",
        "up_proj": "#56B4E9",
        "down_proj": "#7A5195",
    }

    for axis, (group_name, parameters) in zip(axes, PARAMETER_GROUPS.items()):
        for parameter in parameters:
            axis.plot(
                layers,
                [parameter_rms[parameter][layer] for layer in layers],
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=colors[parameter],
                label=parameter.replace("_", " "),
            )
        axis.set_title(group_name)
        axis.set_xlabel("Transformer layer")
        axis.set_xticks(layers)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)

    axes[0].set_ylabel(r"Weight RMS, $\sqrt{\mathrm{mean}(W^2)}$")
    fig.suptitle(f"Matrix parameter RMS across layers\n{model_name}", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args.model, args.local_files_only)
    with (model_dir / "config.json").open() as file:
        config = json.load(file)
    num_layers = config["num_hidden_layers"]

    parameter_rms = compute_rms(find_weight_files(model_dir))
    validate(parameter_rms, num_layers)

    csv_path = args.csv if args.csv is not None else args.output.with_suffix(".csv")
    write_csv(csv_path, parameter_rms, num_layers)
    make_plot(args.output, args.model, parameter_rms, num_layers)
    print(f"Saved plot to {args.output}")
    print(f"Saved values to {csv_path}")


if __name__ == "__main__":
    main()
