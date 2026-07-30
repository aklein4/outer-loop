from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from data.datasets import get_dataset
from models import load_checkpoint
from utils import constants
from utils.import_utils import import_collator


GATE_THRESHOLDS = (0.01, 0.05, 0.95, 0.99)
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
MODULE_RE = re.compile(
    r"^(backbone|output|memory)_layers\.layers\.(\d+)\.state_mechanism"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Slither read/write gates.")
    parser.add_argument("--checkpoint", default="aklein4/slither_alpha-350m")
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument(
        "--data-config",
        default="data/longattn-smollm2.yaml",
    )
    parser.add_argument(
        "--trainer-config",
        default="trainer/slither-med.yaml",
    )
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=constants.LOCAL_DATA_PATH / "slither_gate_analysis",
    )
    return parser.parse_args()


def describe(values: np.ndarray) -> dict[str, float | int]:
    x = values.astype(np.float32, copy=False).reshape(-1)
    qs = np.quantile(x, QUANTILES)
    result: dict[str, float | int] = {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }
    result.update({f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, qs)})
    result.update(
        {
            "frac_lt_0.01": float(np.mean(x < 0.01)),
            "frac_lt_0.05": float(np.mean(x < 0.05)),
            "frac_gt_0.95": float(np.mean(x > 0.95)),
            "frac_gt_0.99": float(np.mean(x > 0.99)),
        }
    )
    result["frac_extreme_0.01"] = result["frac_lt_0.01"] + result["frac_gt_0.99"]
    result["frac_practical_0.05"] = result["frac_lt_0.05"] + result["frac_gt_0.95"]
    return result


def module_metadata(parent_name: str) -> tuple[str, int, str]:
    match = MODULE_RE.match(parent_name)
    if match is None:
        raise ValueError(f"Unexpected state-mechanism path: {parent_name}")
    family = match.group(1)
    local_layer = int(match.group(2))
    return family, local_layer, f"{family}:{local_layer:02d}"


def load_batch(data_config, num_examples: int, device: torch.device):
    dataset = get_dataset(data_config.dataset.url, data_config.dataset.kwargs)
    rows = []
    iterator = iter(dataset)
    for _ in range(num_examples):
        rows.append(next(iterator))
    collator = import_collator(data_config.collator.type)(
        **data_config.collator.kwargs
    )
    batch = collator(rows)["input_ids"].to(device)
    del iterator, dataset, rows
    gc.collect()
    return batch


def register_gate_hooks(model, records, context):
    handles = []
    for name, module in model.named_modules():
        if name.endswith(".state_mechanism.read_gate"):
            gate_type = "read"
            parent = name.removesuffix(".read_gate")
        elif name.endswith(".state_mechanism.writer.write_gate"):
            gate_type = "write"
            parent = name.removesuffix(".writer.write_gate")
        else:
            continue

        family, local_layer, label = module_metadata(parent)

        def hook(_module, _inputs, output, *, gate_type=gate_type,
                 family=family, local_layer=local_layer, label=label):
            gate = torch.sigmoid(output.detach()).to(torch.float16).cpu().numpy()
            records.append(
                {
                    "gate": gate_type,
                    "family": family,
                    "layer": local_layer,
                    "module": label,
                    "chunk": context["chunk"],
                    "values": gate,
                }
            )

        handles.append(module.register_forward_hook(hook))
    return handles


@torch.inference_mode()
def capture(model, tokens, chunk_length: int, use_autocast: bool):
    episodes = list(tokens[:, :-1].split(chunk_length, dim=1))
    if len(episodes) < 2:
        raise ValueError("Gate analysis requires at least two chunks.")

    records: list[dict] = []
    context = {"chunk": -1}
    handles = register_gate_hooks(model, records, context)
    model.init_state(tokens.shape[0], tokens.device)

    previous_mem = None
    started = time.time()
    for chunk_idx, input_ids in enumerate(episodes):
        context["chunk"] = chunk_idx
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            _, new_mem = model(
                input_ids=input_ids,
                mem_states=previous_mem,
                skip_logits=True,
            )
        new_mem = new_mem.float()
        if chunk_idx < len(episodes) - 1:
            model.increment_state(new_mem)
        previous_mem = new_mem
        torch.cuda.synchronize()
        print(
            f"chunk {chunk_idx:02d}/{len(episodes) - 1:02d} "
            f"({input_ids.shape[1]} tokens), elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    for handle in handles:
        handle.remove()
    model.empty_state()
    return records, [int(x.shape[1]) for x in episodes]


def make_event_stats(records, chunk_length: int) -> pd.DataFrame:
    rows = []
    for record in records:
        stats = describe(record["values"])
        rows.append(
            {
                "gate": record["gate"],
                "family": record["family"],
                "layer": record["layer"],
                "module": record["module"],
                "chunk": record["chunk"],
                "sequence_start": record["chunk"] * chunk_length,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def grouped_stats(records, keys: tuple[str, ...]) -> pd.DataFrame:
    groups = defaultdict(list)
    metadata = {}
    for record in records:
        key = tuple(record[k] for k in keys)
        groups[key].append(record["values"].reshape(-1))
        metadata[key] = {k: record[k] for k in keys}
    rows = []
    for key, arrays in groups.items():
        values = np.concatenate(arrays)
        rows.append({**metadata[key], **describe(values)})
    return pd.DataFrame(rows)


def make_head_stats(records) -> pd.DataFrame:
    groups = defaultdict(list)
    metadata = {}
    for record in records:
        for head in range(record["values"].shape[-1]):
            key = (record["gate"], record["module"], head)
            groups[key].append(record["values"][..., head].reshape(-1))
            metadata[key] = {
                "gate": record["gate"],
                "family": record["family"],
                "layer": record["layer"],
                "module": record["module"],
                "head": head,
            }
    return pd.DataFrame(
        [
            {**metadata[key], **describe(np.concatenate(arrays))}
            for key, arrays in groups.items()
        ]
    )


def make_position_stats(records, chunk_length: int, position_bin: int = 32):
    groups = defaultdict(list)
    metadata = {}
    for record in records:
        values = record["values"]
        length = values.shape[1]
        for start in range(0, length, position_bin):
            stop = min(start + position_bin, length)
            key = (record["gate"], record["chunk"], start)
            groups[key].append(values[:, start:stop].reshape(-1))
            metadata[key] = {
                "gate": record["gate"],
                "chunk": record["chunk"],
                "within_chunk_start": start,
                "within_chunk_center": (start + stop - 1) / 2,
                "sequence_center": record["chunk"] * chunk_length
                + (start + stop - 1) / 2,
            }
    return pd.DataFrame(
        [
            {**metadata[key], **describe(np.concatenate(arrays))}
            for key, arrays in groups.items()
        ]
    )


def plot_distributions(records, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    bins = np.linspace(0, 1, 101)
    for ax, gate in zip(axes, ("read", "write")):
        all_values = np.concatenate(
            [r["values"].reshape(-1) for r in records if r["gate"] == gate]
        )
        ax.hist(all_values, bins=bins, density=True, color="#4472c4", alpha=0.85)
        ax.axvline(0.05, color="#c44e52", linestyle="--", linewidth=1)
        ax.axvline(0.95, color="#c44e52", linestyle="--", linewidth=1)
        ax.set(title=f"{gate.capitalize()} gate", xlabel="sigmoid gate value")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_chunk_trends(event_stats: pd.DataFrame, output: Path):
    aggregated = (
        event_stats.groupby(["gate", "chunk"])
        .apply(
            lambda x: pd.Series(
                {
                    "mean": np.average(x["mean"], weights=x["count"]),
                    "practical": np.average(
                        x["frac_practical_0.05"], weights=x["count"]
                    ),
                    "extreme": np.average(
                        x["frac_extreme_0.01"], weights=x["count"]
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
    colors = {"read": "#4472c4", "write": "#dd8452"}
    for gate, frame in aggregated.groupby("gate"):
        axes[0].plot(
            frame["chunk"], frame["mean"], marker=".", label=gate, color=colors[gate]
        )
        axes[1].plot(
            frame["chunk"],
            100 * frame["practical"],
            marker=".",
            label=f"{gate}: <.05 or >.95",
            color=colors[gate],
        )
        axes[1].plot(
            frame["chunk"],
            100 * frame["extreme"],
            linestyle="--",
            label=f"{gate}: <.01 or >.99",
            color=colors[gate],
        )
    axes[0].set(ylabel="mean gate value", title="Gate behavior by chunk")
    axes[1].set(xlabel="chunk index", ylabel="saturated activations (%)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_within_chunk(position_stats: pd.DataFrame, output: Path):
    aggregated = (
        position_stats.groupby(["gate", "within_chunk_start"])
        .apply(
            lambda x: pd.Series(
                {
                    "mean": np.average(x["mean"], weights=x["count"]),
                    "practical": np.average(
                        x["frac_practical_0.05"], weights=x["count"]
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for gate, frame in aggregated.groupby("gate"):
        axes[0].plot(frame["within_chunk_start"], frame["mean"], label=gate)
        axes[1].plot(
            frame["within_chunk_start"],
            100 * frame["practical"],
            label=gate,
        )
    axes[0].set(xlabel="position within 1,024-token chunk", ylabel="mean gate")
    axes[1].set(
        xlabel="position within 1,024-token chunk",
        ylabel="activations <.05 or >.95 (%)",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_layer_heatmaps(event_stats: pd.DataFrame, output: Path):
    module_order = list(
        dict.fromkeys(
            event_stats.sort_values(["family", "layer"])["module"].tolist()
        )
    )
    # Put model execution order ahead of alphabetic family order.
    module_order = [
        f"{family}:{layer:02d}"
        for family, count in (("backbone", 18), ("output", 2), ("memory", 4))
        for layer in range(count)
        if f"{family}:{layer:02d}" in module_order
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, gate in zip(axes, ("read", "write")):
        frame = event_stats[event_stats.gate == gate]
        pivot = frame.pivot(index="module", columns="chunk", values="mean").reindex(
            module_order
        )
        image = ax.imshow(
            pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis"
        )
        ax.set_yticks(range(len(module_order)), module_order, fontsize=7)
        ax.set_ylabel("state mechanism")
        ax.set_title(f"{gate.capitalize()} gate mean")
        fig.colorbar(image, ax=ax, label="mean gate")
    axes[-1].set_xlabel("chunk index")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_head_std_histograms(
    head_stats: pd.DataFrame,
    output: Path,
    low_variance_threshold: float = 0.1,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    bins = np.linspace(0, 0.32, 33)
    for ax, gate in zip(axes, ("read", "write")):
        values = head_stats.loc[head_stats.gate == gate, "std"]
        low_count = int((values <= low_variance_threshold).sum())
        ax.hist(values, bins=bins, color="#4472c4", alpha=0.85)
        ax.axvline(
            low_variance_threshold,
            color="#c44e52",
            linestyle="--",
            linewidth=1.5,
            label=f"low variance ≤ {low_variance_threshold:g}",
        )
        ax.set(
            title=f"{gate.capitalize()} heads ({low_count}/{len(values)} low variance)",
            xlabel="post-sigmoid standard deviation",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("number of gate heads")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This analysis requires the requested CUDA GPU.")

    data_config = OmegaConf.load(constants.CONFIG_PATH(args.data_config))
    trainer_config = OmegaConf.load(constants.CONFIG_PATH(args.trainer_config))
    output = args.output_dir / (
        f"{args.checkpoint.replace('/', '--')}_step={args.checkpoint_step}"
        f"_n={args.num_examples}"
    )
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    tokens = load_batch(data_config, args.num_examples, device)
    model = load_checkpoint(
        args.checkpoint,
        args.checkpoint_step,
        attention_kernel="gpu_flash_attention",
    )
    # Match training: fp32 parameters/state, with bf16 autocast around forward.
    # State writes happen outside autocast in SlitherTrainer.go_forward.
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records, episode_lengths = capture(
        model,
        tokens,
        model.chunk_length,
        bool(trainer_config.use_autocast),
    )

    event_stats = make_event_stats(records, model.chunk_length)
    module_stats = grouped_stats(records, ("gate", "family", "layer", "module"))
    family_stats = grouped_stats(records, ("gate", "family"))
    gate_stats = grouped_stats(records, ("gate",))
    head_stats = make_head_stats(records)
    position_stats = make_position_stats(records, model.chunk_length)

    event_stats.to_csv(output / "chunk_module_stats.csv", index=False)
    module_stats.to_csv(output / "module_stats.csv", index=False)
    family_stats.to_csv(output / "family_stats.csv", index=False)
    head_stats.to_csv(output / "head_stats.csv", index=False)
    head_stats[head_stats["std"] <= 0.1].sort_values(
        ["gate", "family", "layer", "head"]
    ).to_csv(output / "low_variance_heads_std_le_0.1.csv", index=False)
    position_stats.to_csv(output / "position_bin_stats.csv", index=False)
    plot_distributions(records, output / "gate_distributions.png")
    plot_chunk_trends(event_stats, output / "chunk_trends.png")
    plot_within_chunk(position_stats, output / "within_chunk_trends.png")
    plot_layer_heatmaps(event_stats, output / "layer_chunk_heatmaps.png")
    plot_head_std_histograms(head_stats, output / "head_std_histograms.png")

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "data_config": str(constants.CONFIG_PATH(args.data_config)),
        "dataset": data_config.dataset.url,
        "num_examples": args.num_examples,
        "configured_sequence_length": int(
            data_config.collator.kwargs.sequence_length
        ),
        "trainer_config": str(constants.CONFIG_PATH(args.trainer_config)),
        "autocast": bool(trainer_config.use_autocast),
        "autocast_dtype": "bfloat16",
        "device": torch.cuda.get_device_name(),
        "attention_kernel": "gpu_flash_attention",
        "chunk_length": int(model.chunk_length),
        "episode_lengths": episode_lengths,
        "num_state_mechanisms": int(module_stats["module"].nunique()),
        "num_state_heads": int(model.config.num_state_heads),
        "gate_statistics": {
            row["gate"]: {
                key: value
                for key, value in row.items()
                if key != "gate"
            }
            for row in gate_stats.to_dict(orient="records")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote analysis to {output}", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    # Streaming pyarrow occasionally aborts during interpreter teardown after an
    # intentionally short read; all files are closed by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
