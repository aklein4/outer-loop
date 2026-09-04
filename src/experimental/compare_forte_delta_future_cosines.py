"""Compare step-local Forte-delta updates with current and future raw gradients.

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
RIGHT = ("raw_G_32", "G_future_32", "kernel_G_32")
DIRECT_RIGHT = ("raw_G_32", "G_future_32")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cosine_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xf, yf = x.float().flatten(1), y.float().flatten(1)
    return ((xf * yf).sum(1) / (xf.norm(dim=1) * yf.norm(dim=1)).clamp_min(1e-30)).clamp(-1, 1)


class CosineCollector:
    def __init__(self, model, target_episode: int):
        self.model = model
        self.target_episode = target_episode
        self.episode = -1
        self.phase = ""
        self.original = forte._get_G
        self.module_by_weight = {
            module.down_fast.weight.data_ptr(): (layer, module)
            for layer, module in enumerate(model.fast_modules())
        }
        self.rows: list[dict] = []
        self.matrices: dict[tuple[int, int, str], torch.Tensor] = {}
        self.kernel_sums: dict[int, torch.Tensor] = {}

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
        if self.phase == "kernel":
            layer, _ = self.module_by_weight[down_weight.data_ptr()]
            value = raw_g.detach().float()
            if layer not in self.kernel_sums:
                self.kernel_sums[layer] = value.clone()
            else:
                self.kernel_sums[layer].add_(value)
            return raw_g, total
        if self.phase != "second" or self.episode != self.target_episode:
            return raw_g, total

        layer, module = self.module_by_weight[down_weight.data_ptr()]
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
            offset = total.detach().float() - no_offset
            # This is the exact tensor used by ForteFastWeightFunction.backward
            # in local_loss = (future_grad * update).sum().
            future = module.grad_buffer.detach().float() - raw_g.detach().float()
            tensors = {
                "raw_G_32": raw_g.detach().float(),
                "no_offset_update_32": no_offset,
                "offset_32": offset,
                "total_delta_32": total.detach().float(),
                "G_future_32": future,
            }
            for left in LEFT:
                for right in DIRECT_RIGHT:
                    values = cosine_rows(tensors[left], tensors[right])
                    for trajectory, value in enumerate(values):
                        self.rows.append({
                            "trajectory": trajectory,
                            "layer": layer,
                            "left": left,
                            "right": right,
                            "cosine": value.item(),
                            "left_norm": tensors[left][trajectory].norm().item(),
                            "right_norm": tensors[right][trajectory].norm().item(),
                        })
            for trajectory in range(raw_g.shape[0]):
                for name, tensor in tensors.items():
                    self.matrices[trajectory, layer, name] = tensor[trajectory].flatten().cpu()
        return raw_g, total

    def finalize_sum(self, name: str, sums: dict[int, torch.Tensor]) -> None:
        for layer, summed_gpu in sums.items():
            summed = summed_gpu.cpu()
            for trajectory in range(summed.shape[0]):
                summed_row = summed[trajectory].flatten()
                self.matrices[trajectory, layer, name] = summed_row
                for left in LEFT:
                    value = cosine_rows(
                        self.matrices[trajectory, layer, left][None],
                        summed_row[None],
                    )[0]
                    self.rows.append({
                        "trajectory": trajectory,
                        "layer": layer,
                        "left": left,
                        "right": name,
                        "cosine": value.item(),
                        "left_norm": self.matrices[trajectory, layer, left].norm().item(),
                        "right_norm": summed_row.norm().item(),
                    })


def loss_and_grad(model, states, ids, assistant, valid, iterations: int, aux_weight: float):
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
    labels = labels.reshape(-1, iterations)
    weights = weights.reshape(-1, iterations)
    for iteration in range(iterations):
        logits = model.lm_head(leaf[:, iteration]).float()
        raw = F.cross_entropy(logits, labels[:, iteration].contiguous(), reduction="none")
        (raw * weights[:, iteration]).sum().backward()
    return leaf.grad.reshape_as(states).detach().to(states.dtype)


def first_pass(model, ids, assistant, valid, iterations: int, aux_weight: float) -> None:
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
        grad = loss_and_grad(model, states, ids, assistant, valid, iterations, aux_weight)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(ForteMode.TRAIN_FIRST)


def second_pass(model, ids, assistant, valid, iterations: int, aux_weight: float) -> None:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        double_ids = torch.repeat_interleave(ids, 2, dim=0)
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, valid)
        hidden = model.forward_backbone(
            double_ids, mode=ForteMode.TRAIN_SECOND,
            embeddings=embeddings, embedding_mask=valid,
        )
        states = model.forward_lm_states(
            hidden, mode=ForteMode.TRAIN_SECOND, logits_to_keep=slice(0, -1),
            embeddings=embeddings, embedding_mask=valid,
        )[::2]
        grad = loss_and_grad(model, states, ids, assistant, valid, iterations, aux_weight)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(ForteMode.TRAIN_SECOND)


def kernel_pass(model, ids, assistant, valid, iterations: int, aux_weight: float) -> None:
    """Measure direct raw G without changing the frozen recurrent state."""
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
        grad = loss_and_grad(model, states, ids, assistant, valid, iterations, aux_weight)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad():
        for module in model.fast_modules():
            module.grad_buffer.grad.zero_()
            module.state.grad.zero_()


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["left"], row["right"]].append(row)
    output = []
    for left in LEFT:
        for right in RIGHT:
            selected = grouped[left, right]
            values = np.asarray([row["cosine"] for row in selected])
            trajectory_means = np.asarray([
                np.mean([row["cosine"] for row in selected if row["trajectory"] == trajectory])
                for trajectory in sorted({row["trajectory"] for row in selected})
            ])
            se = trajectory_means.std(ddof=1) / math.sqrt(len(trajectory_means))
            output.append({
                "left": left,
                "right": right,
                "observations": len(values),
                "trajectories": len(trajectory_means),
                "layer_balanced_mean": values.mean(),
                "median": np.median(values),
                "trajectory_mean_std": trajectory_means.std(ddof=1),
                "ci95_low": trajectory_means.mean() - 1.96 * se,
                "ci95_high": trajectory_means.mean() + 1.96 * se,
                "fraction_positive": np.mean(values > 0),
            })
    return output


def stacked_rows(collector: CosineCollector) -> list[dict]:
    rows = []
    trajectories = sorted({key[0] for key in collector.matrices})
    layers = sorted({key[1] for key in collector.matrices})
    for trajectory in trajectories:
        names = set(LEFT) | set(RIGHT)
        joined = {
            name: torch.cat([collector.matrices[trajectory, layer, name] for layer in layers])
            for name in names
        }
        for left in LEFT:
            for right in RIGHT:
                x, y = joined[left].double(), joined[right].double()
                rows.append({
                    "trajectory": trajectory,
                    "left": left,
                    "right": right,
                    "cosine": ((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-300)).item(),
                })
    return rows


def summarize_stacked(rows: list[dict]) -> list[dict]:
    output = []
    for left in LEFT:
        for right in RIGHT:
            values = np.asarray([
                row["cosine"] for row in rows
                if row["left"] == left and row["right"] == right
            ])
            se = values.std(ddof=1) / math.sqrt(len(values))
            output.append({
                "left": left,
                "right": right,
                "trajectories": len(values),
                "stacked_layer_mean": values.mean(),
                "median": np.median(values),
                "ci95_low": values.mean() - 1.96 * se,
                "ci95_high": values.mean() + 1.96 * se,
            })
    return output


def plot(summary: list[dict], path: Path) -> None:
    values = np.asarray([
        [next(row["layer_balanced_mean"] for row in summary
              if row["left"] == left and row["right"] == right) for right in RIGHT]
        for left in LEFT
    ])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center",
                    color="white" if abs(values[i, j]) > .5 else "black")
    ax.set_xticks(range(len(RIGHT)), [name.replace("_", "\n") for name in RIGHT])
    ax.set_yticks(range(len(LEFT)), [name.replace("_", " ") for name in LEFT])
    ax.set_title("Update 32: layer-balanced Frobenius cosine")
    fig.colorbar(image, ax=ax, label="cosine similarity")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="aklein4/horizon-v2_forte-delta")
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--update-number", type=int, default=32,
                        help="One-based recurrent update number to inspect.")
    parser.add_argument("--data-config", type=Path,
                        default=SRC / "configs/data/300k-horizons-llama3.yaml")
    parser.add_argument("--aux-loss-weight", type=float, default=.1)
    parser.add_argument("--num-logit-iterations", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        default=SRC / "local_data/horizon_v2_forte_delta_step1000_g32_cosines")
    args = parser.parse_args()
    if not 1 <= args.update_number < args.chunks:
        raise ValueError("update-number must leave at least one future chunk")
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.data_config)
    stream = datasets.load_dataset(config.dataset.url, **config.dataset.kwargs)
    iterator = iter(stream)
    raw = [next(iterator) for _ in range(args.trajectories)]
    collator_kwargs = OmegaConf.to_container(config.collator.kwargs, resolve=True)
    collator_kwargs["cluster_length"] = args.chunks
    batch = HorizonCollator(**collator_kwargs)(raw)
    ids, assistant, valid = (
        batch[key].cuda() for key in ("input_ids", "assistant_mask", "attention_mask")
    )

    model = load_checkpoint(args.checkpoint, args.step, attention_kernel=None).cuda().eval()
    model.lm_head.to(torch.bfloat16)
    enable_gradient_checkpointing(model, True)
    model.train()
    model.init_state(args.trajectories, torch.device("cuda"))
    collector = CosineCollector(model, args.update_number - 1)
    collector.install()
    try:
        for episode in range(args.chunks):
            collector.phase, collector.episode = "first", episode
            first_pass(
                model, ids[:, episode], assistant[:, episode], valid[:, episode],
                args.num_logit_iterations, args.aux_loss_weight,
            )
            print(f"first pass {episode + 1}/{args.chunks}", flush=True)
        model.finalize_state()
        model.zero_grad(set_to_none=False)
        for episode in range(args.update_number):
            if episode == args.update_number - 1:
                pre_update_state = [
                    module.state.detach().clone() for module in model.fast_modules()
                ]
            collector.phase, collector.episode = "second", episode
            second_pass(
                model, ids[:, episode], assistant[:, episode], valid[:, episode],
                args.num_logit_iterations, args.aux_loss_weight,
            )
            print(f"second pass {episode + 1}/{args.update_number}", flush=True)
        with torch.no_grad():
            for module, state in zip(model.fast_modules(), pre_update_state):
                module.state.copy_(state)
                module.state.grad.zero_()
                module.grad_buffer.grad.zero_()
        for episode in range(args.update_number, args.chunks):
            collector.phase, collector.episode = "kernel", episode
            kernel_pass(
                model, ids[:, episode], assistant[:, episode], valid[:, episode],
                args.num_logit_iterations, args.aux_loss_weight,
            )
            print(f"kernel future {episode + 1}/{args.chunks}", flush=True)
        collector.finalize_sum("kernel_G_32", collector.kernel_sums)
    finally:
        collector.remove()

    rows = collector.rows
    layer_summary = summarize(rows)
    global_rows = stacked_rows(collector)
    global_summary = summarize_stacked(global_rows)
    by_layer = []
    for layer in range(len(model.fast_modules())):
        for left in LEFT:
            for right in RIGHT:
                values = [row["cosine"] for row in rows if row["layer"] == layer
                          and row["left"] == left and row["right"] == right]
                by_layer.append({
                    "layer": layer, "left": left, "right": right,
                    "mean": np.mean(values), "median": np.median(values),
                    "min": np.min(values), "max": np.max(values),
                })
    write_csv(args.output_dir / "cosines_by_trajectory_layer.csv", rows)
    write_csv(args.output_dir / "cosine_summary_layer_balanced.csv", layer_summary)
    write_csv(args.output_dir / "cosines_stacked_layers_by_trajectory.csv", global_rows)
    write_csv(args.output_dir / "cosine_summary_stacked_layers.csv", global_summary)
    write_csv(args.output_dir / "cosines_by_layer.csv", by_layer)
    plot(layer_summary, args.output_dir / "cosine_matrix.png")
    metadata = {
        "checkpoint": args.checkpoint,
        "step": args.step,
        "dataset": config.dataset.url,
        "trajectories": args.trajectories,
        "chunks": args.chunks,
        "update_number_one_based": args.update_number,
        "episode_index_zero_based": args.update_number - 1,
        "G_future_definition": "grad_buffer immediately before update 32 minus raw_G_32; this is the exact future_grad used in local_loss=(future_grad*update).sum()",
        "kernel_G_definition": "sum of raw G for one-based future episodes 33-64, each evaluated independently at the recurrent state immediately before update 32, with no state updates between examples",
        "offset_definition": "realized norm-capped contribution total_delta_32 - no_offset_update_32",
        "cosine_definition": "signed Frobenius cosine after flattening each fast-weight matrix",
        "aux_loss_weight": args.aux_loss_weight,
        "num_logit_iterations": args.num_logit_iterations,
        "torch_xla_used": False,
        "cuda_device": torch.cuda.get_device_name(),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"layer_balanced": layer_summary, "stacked": global_summary}, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
