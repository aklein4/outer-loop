"""First-pass counterfactual future gradient after omitting recurrent update 32.

CUDA/PyTorch only. This intentionally does not import or require torch-xla.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator  # noqa: E402
from models import load_checkpoint  # noqa: E402
import models.forte as forte  # noqa: E402
from models.forte import ForteMode  # noqa: E402
from utils.torch_modules import enable_gradient_checkpointing  # noqa: E402
from utils.torch_utils import unit_softplus  # noqa: E402

LEFT = ("raw_G_32", "no_offset_update_32", "offset_32", "total_delta_32")
RIGHT = "counterfactual_G_future_32"


def cosine_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xf, yf = x.float().flatten(1), y.float().flatten(1)
    value = (xf * yf).sum(1) / (xf.norm(dim=1) * yf.norm(dim=1)).clamp_min(1e-30)
    return value.clamp(-1, 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Collector:
    def __init__(self, model, target_episode: int, target_offset_scale: float = 1.0,
                 future_name: str = RIGHT):
        self.target_episode = target_episode
        self.target_offset_scale = target_offset_scale
        self.future_name = future_name
        self.episode = -1
        self.phase = ""
        self.original = forte._get_G
        self.module_by_weight = {
            module.down_fast.weight.data_ptr(): (layer, module)
            for layer, module in enumerate(model.fast_modules())
        }
        self.target: dict[tuple[int, int, str], torch.Tensor] = {}
        self.future_sums: dict[int, torch.Tensor] = {}

    def install(self) -> None:
        forte._get_G = self.get_g

    def remove(self) -> None:
        forte._get_G = self.original

    def get_g(self, activations, output_grad, down_weight, valid_mask, *dynamic_args,
              eps: float, offset_alpha: float):
        raw_g, total = self.original(
            activations, output_grad, down_weight, valid_mask, *dynamic_args,
            eps=eps, offset_alpha=offset_alpha,
        )
        layer, _ = self.module_by_weight[down_weight.data_ptr()]
        if self.phase == "future":
            value = raw_g.detach().float()
            if layer not in self.future_sums:
                self.future_sums[layer] = value.clone()
            else:
                self.future_sums[layer].add_(value)
            return raw_g, total
        if self.phase != "target" or self.episode != self.target_episode:
            return raw_g, total

        with torch.no_grad():
            _, _, lr, _, gradient_logits, _ = [value.detach().float() for value in dynamic_args]
            mask = valid_mask.bool()[..., None].float()
            count = mask.sum(-2, keepdim=True).clamp_min(1)
            a = activations.detach().float() * mask
            g = F.linear(
                output_grad.detach().float() * mask,
                down_weight.detach().float().T,
            )
            a_norm = a * torch.rsqrt(a.square().sum(-2, keepdim=True) / count + eps**2)
            g_norm = g * torch.rsqrt(g.square().sum(-2, keepdim=True) / count + eps**2)
            no_offset = -torch.einsum(
                "blo,bli->boi",
                forte.gate_heads(g_norm, unit_softplus(gradient_logits)),
                a_norm,
            ) * lr
            tensors = {
                "raw_G_32": raw_g.detach().float(),
                "no_offset_update_32": no_offset,
                "offset_32": total.detach().float() - no_offset,
                "total_delta_32": total.detach().float(),
            }
            for trajectory in range(raw_g.shape[0]):
                for name, tensor in tensors.items():
                    self.target[trajectory, layer, name] = tensor[trajectory].flatten().cpu()
        return raw_g, no_offset + self.target_offset_scale * tensors["offset_32"]

    def results(self) -> tuple[list[dict], list[dict]]:
        rows = []
        for layer, future_gpu in self.future_sums.items():
            future = future_gpu.cpu()
            for trajectory in range(future.shape[0]):
                future_row = future[trajectory].flatten()
                self.target[trajectory, layer, self.future_name] = future_row
                for left in LEFT:
                    x = self.target[trajectory, layer, left]
                    rows.append({
                        "trajectory": trajectory,
                        "layer": layer,
                        "left": left,
                        "right": self.future_name,
                        "cosine": cosine_rows(x[None], future_row[None])[0].item(),
                        "left_norm": x.norm().item(),
                        "right_norm": future_row.norm().item(),
                    })
        stacked = []
        trajectories = sorted({key[0] for key in self.target})
        layers = sorted(self.future_sums)
        for trajectory in trajectories:
            future = torch.cat([
                self.target[trajectory, layer, self.future_name] for layer in layers
            ]).double()
            for left in LEFT:
                x = torch.cat([self.target[trajectory, layer, left] for layer in layers]).double()
                stacked.append({
                    "trajectory": trajectory,
                    "left": left,
                    "right": self.future_name,
                    "cosine": ((x * future).sum() / (x.norm() * future.norm()).clamp_min(1e-300)).item(),
                })
        return rows, stacked


def loss_gradient(model, states, ids, assistant, valid, iterations: int, aux_weight: float):
    batch = states.shape[0]
    labels = ids[:, 1:]
    assistant = assistant[:, 1:].float()
    valid = valid[:, 1:].float()
    nonassistant = valid - assistant
    weights = (
        assistant / assistant.sum(-1, keepdim=True).clamp_min(1)
        + aux_weight * nonassistant / nonassistant.sum(-1, keepdim=True).clamp_min(1)
    ) / batch
    leaf = states.detach().reshape(-1, iterations, states.shape[-1]).requires_grad_(True)
    labels, weights = labels.reshape(-1, iterations), weights.reshape(-1, iterations)
    for iteration in range(iterations):
        logits = model.lm_head(leaf[:, iteration]).float()
        raw = F.cross_entropy(logits, labels[:, iteration].contiguous(), reduction="none")
        (raw * weights[:, iteration]).sum().backward()
    return leaf.grad.reshape_as(states).detach().to(states.dtype)


def step(model, ids, assistant, valid, iterations: int, aux_weight: float,
         apply_update: bool) -> None:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, valid).detach()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_backbone(
            ids, mode=ForteMode.TRAIN_FIRST, embeddings=embeddings, embedding_mask=valid,
        )
        states = model.forward_lm_states(
            hidden, mode=ForteMode.TRAIN_FIRST, logits_to_keep=slice(0, -1),
            embeddings=embeddings, embedding_mask=valid,
        )
        grad = loss_gradient(model, states, ids, assistant, valid, iterations, aux_weight)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad():
        if apply_update:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.update_state(ForteMode.TRAIN_FIRST)
        else:
            for module in model.fast_modules():
                module.state.grad.zero_()
                module.grad_buffer.grad.zero_()


def summaries(rows: list[dict], stacked: list[dict]) -> tuple[list[dict], list[dict]]:
    layer_summary, stacked_summary = [], []
    right = rows[0]["right"]
    for left in LEFT:
        selected = [row for row in rows if row["left"] == left]
        values = np.asarray([row["cosine"] for row in selected])
        trajectory_means = np.asarray([
            np.mean([row["cosine"] for row in selected if row["trajectory"] == trajectory])
            for trajectory in range(16)
        ])
        se = trajectory_means.std(ddof=1) / math.sqrt(len(trajectory_means))
        layer_summary.append({
            "left": left, "right": right, "observations": len(values),
            "layer_balanced_mean": values.mean(), "median": np.median(values),
            "ci95_low": trajectory_means.mean() - 1.96 * se,
            "ci95_high": trajectory_means.mean() + 1.96 * se,
            "fraction_positive": np.mean(values > 0),
        })
        global_values = np.asarray([row["cosine"] for row in stacked if row["left"] == left])
        global_se = global_values.std(ddof=1) / math.sqrt(len(global_values))
        stacked_summary.append({
            "left": left, "right": right, "trajectories": len(global_values),
            "stacked_layer_mean": global_values.mean(), "median": np.median(global_values),
            "ci95_low": global_values.mean() - 1.96 * global_se,
            "ci95_high": global_values.mean() + 1.96 * global_se,
        })
    return layer_summary, stacked_summary


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
                        "horizon_v2_forte_delta_step1000_counterfactual_gfuture32")
    args = parser.parse_args()
    assert torch.cuda.is_available() and 1 <= args.update_number < args.chunks
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
        end = min(start + args.batch_size, args.trajectories)
        ids, assistant, valid = (
            batch[key][start:end].cuda()
            for key in ("input_ids", "assistant_mask", "attention_mask")
        )
        model.init_state(end - start, torch.device("cuda"))
        collector = Collector(model, args.update_number - 1)
        collector.install()
        try:
            for episode in range(args.chunks):
                if episode < args.update_number - 1:
                    collector.phase = "prefix"
                    apply = True
                elif episode == args.update_number - 1:
                    collector.phase = "target"
                    apply = False
                else:
                    collector.phase = "future"
                    apply = True
                collector.episode = episode
                step(
                    model, ids[:, episode], assistant[:, episode], valid[:, episode],
                    args.num_logit_iterations, args.aux_loss_weight, apply,
                )
                print(f"batch {start // args.batch_size + 1}/"
                      f"{math.ceil(args.trajectories / args.batch_size)} "
                      f"first-pass counterfactual {episode + 1}/{args.chunks} "
                      f"({'apply' if apply else 'skip'} update)", flush=True)
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
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(range(len(LEFT)), values, color="#4472c4")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(len(LEFT)), [name.replace("_", "\n") for name in LEFT])
    ax.set_ylabel("cosine vs counterfactual G_future_32")
    fig.savefig(args.output_dir / "counterfactual_cosines.png", dpi=180)
    plt.close(fig)
    metadata = {
        "checkpoint": args.checkpoint, "step": args.step,
        "dataset": config.dataset.url, "trajectories": args.trajectories,
        "batch_size": args.batch_size,
        "chunks": args.chunks, "update_number_one_based": args.update_number,
        "construction": "single TRAIN_FIRST rollout: apply updates 1-31, compute but omit update 32, then apply updates 33-64 normally and sum their raw G matrices",
        "counterfactual_G_future_definition": "sum of future raw G for one-based episodes 33-64 along the recurrent trajectory in which full delta 32 is omitted",
        "offset_definition": "realized norm-capped contribution total_delta_32 - no_offset_update_32",
        "cosine_definition": "signed Frobenius cosine after flattening each fast-weight matrix",
        "torch_xla_used": False, "cuda_device": torch.cuda.get_device_name(),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"layer_balanced": layer_summary, "stacked": stacked_summary}, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
