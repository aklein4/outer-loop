"""Compare update-32 components to a future kernel at the no-offset state.

CUDA/PyTorch only. This intentionally does not import or require torch-xla.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator  # noqa: E402
from models import load_checkpoint  # noqa: E402
from models.forte import ForteMode  # noqa: E402
from utils.torch_modules import enable_gradient_checkpointing  # noqa: E402
from compare_forte_delta_counterfactual_future import (  # noqa: E402
    Collector, LEFT, step, summaries, write_csv,
)

RIGHT = "kernel_G_after_no_offset_32"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="aklein4/horizon-v2_forte-delta")
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--update-number", type=int, default=32)
    parser.add_argument("--data-config", type=Path,
                        default=SRC / "configs/data/300k-horizons-llama3.yaml")
    parser.add_argument("--aux-loss-weight", type=float, default=.1)
    parser.add_argument("--num-logit-iterations", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=SRC / "local_data/"
                        "horizon_v2_forte_delta_step1000_no_offset32_kernel")
    args = parser.parse_args()
    assert torch.cuda.is_available() and 1 <= args.update_number < args.chunks
    assert args.trajectories % args.batch_size == 0
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.data_config)
    stream = datasets.load_dataset(config.dataset.url, **config.dataset.kwargs)
    iterator = iter(stream)
    raw = [next(iterator) for _ in range(args.trajectories)]
    kwargs = OmegaConf.to_container(config.collator.kwargs, resolve=True)
    kwargs["cluster_length"] = args.chunks
    batch = HorizonCollator(**kwargs)(raw)
    model = load_checkpoint(args.checkpoint, args.step, attention_kernel=None).cuda().eval()
    model.lm_head.to(torch.bfloat16)
    enable_gradient_checkpointing(model, True)
    model.train()

    rows, stacked = [], []
    for start in range(0, args.trajectories, args.batch_size):
        end = start + args.batch_size
        ids, assistant, valid = (
            batch[key][start:end].cuda()
            for key in ("input_ids", "assistant_mask", "attention_mask")
        )
        model.init_state(args.batch_size, torch.device("cuda"))
        collector = Collector(
            model, args.update_number - 1,
            target_offset_scale=0.0,
            future_name=RIGHT,
        )
        collector.install()
        try:
            for episode in range(args.chunks):
                collector.episode = episode
                if episode < args.update_number - 1:
                    collector.phase, apply = "prefix", True
                elif episode == args.update_number - 1:
                    # Collector returns base + 0 * offset, and this base-only
                    # update is written into state.
                    collector.phase, apply = "target", True
                else:
                    # Measure every future raw G at the same post-base state.
                    collector.phase, apply = "future", False
                step(
                    model, ids[:, episode], assistant[:, episode], valid[:, episode],
                    args.num_logit_iterations, args.aux_loss_weight, apply,
                )
                print(f"batch {start // args.batch_size + 1}/"
                      f"{args.trajectories // args.batch_size} episode "
                      f"{episode + 1}/{args.chunks} "
                      f"({'apply base only' if episode == args.update_number - 1 else 'apply' if apply else 'freeze'})",
                      flush=True)
        finally:
            collector.remove()
        group_rows, group_stacked = collector.results()
        for row in group_rows + group_stacked:
            row["trajectory"] += start
        rows.extend(group_rows)
        stacked.extend(group_stacked)
        del ids, assistant, valid, collector
        torch.cuda.empty_cache()

    layer_summary, stacked_summary = summaries(rows, stacked)
    by_layer = []
    for layer in range(len(model.fast_modules())):
        for left in LEFT:
            values = [row["cosine"] for row in rows
                      if row["layer"] == layer and row["left"] == left]
            by_layer.append({
                "layer": layer, "left": left, "right": RIGHT,
                "mean": np.mean(values), "median": np.median(values),
                "min": np.min(values), "max": np.max(values),
            })
    write_csv(args.output_dir / "cosines_by_trajectory_layer.csv", rows)
    write_csv(args.output_dir / "cosine_summary_layer_balanced.csv", layer_summary)
    write_csv(args.output_dir / "cosines_stacked_layers_by_trajectory.csv", stacked)
    write_csv(args.output_dir / "cosine_summary_stacked_layers.csv", stacked_summary)
    write_csv(args.output_dir / "cosines_by_layer.csv", by_layer)

    values = [row["layer_balanced_mean"] for row in layer_summary]
    lows = [row["ci95_low"] for row in layer_summary]
    highs = [row["ci95_high"] for row in layer_summary]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar(range(len(LEFT)), values, color="#4472c4")
    ax.errorbar(range(len(LEFT)), values,
                yerr=[np.asarray(values) - np.asarray(lows),
                      np.asarray(highs) - np.asarray(values)],
                fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(len(LEFT)), [name.replace("_", "\n") for name in LEFT])
    ax.set_ylabel(f"cosine vs {RIGHT}")
    ax.set_title("Update 32 vs future kernel at the base-only state")
    fig.savefig(args.output_dir / "no_offset32_kernel_cosines.png", dpi=180)
    plt.close(fig)

    metadata = {
        "checkpoint": args.checkpoint, "step": args.step,
        "dataset": config.dataset.url, "trajectories": args.trajectories,
        "batch_size": args.batch_size, "chunks": args.chunks,
        "update_number_one_based": args.update_number,
        "construction": "single TRAIN_FIRST rollout: apply full updates 1-31, apply only the no-offset preconditioned/gated/LR-scaled update at 32, then freeze state and sum raw G from episodes 33-64",
        "kernel_definition": "sum of independently evaluated future raw G matrices at the state after the no-offset update 32; no updates are applied after 32",
        "offset_definition": "the normal realized norm-capped contribution omitted from state at update 32",
        "cosine_definition": "signed Frobenius cosine after flattening each fast-weight matrix",
        "torch_xla_used": False, "cuda_device": torch.cuda.get_device_name(),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"layer_balanced": layer_summary, "stacked": stacked_summary}, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
