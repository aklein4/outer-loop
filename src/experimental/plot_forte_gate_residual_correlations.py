"""Remove common modes from Forte gates and correlate residual dimensions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import plot_forte_gate_dimension_correlations as base

from collators.horizon import HorizonCollator
from models import load_checkpoint
import models.forte as forte_module
from utils import constants
from utils.torch_modules import enable_gradient_checkpointing


LAYERS = (7, 14)
GATES = ("a", "g")
REPRESENTATIONS = ("clr_post_sigmoid", "centered_logits")
GROUPS = ("assistant", "non_assistant", "all_valid")


class ResidualCollector:
    def __init__(self, model, log_eps=1e-8):
        self.original = forte_module._get_G
        self.log_eps = log_eps
        self.layer_by_weight = {
            module.down_fast.weight.data_ptr(): layer
            for layer, module in enumerate(model.fast_modules())
        }
        self.token_masks = None
        self.values = {
            (layer, gate, representation, group): []
            for layer in LAYERS for gate in GATES
            for representation in REPRESENTATIONS
            for group in ("assistant", "non_assistant")
        }

    def install(self):
        forte_module._get_G = self.get_g

    def remove(self):
        forte_module._get_G = self.original

    def set_masks(self, assistant, attention):
        valid = attention.bool()
        assistant = assistant.bool() & valid
        self.token_masks = {
            "assistant": assistant,
            "non_assistant": valid & ~assistant,
        }

    def get_g(self, activations, output_grad, down_weight,
              activation_gate_logits, gradient_gate_logits, eps):
        layer = self.layer_by_weight[down_weight.data_ptr()]
        if layer in LAYERS:
            logits = {
                "a": activation_gate_logits.detach().float(),
                "g": gradient_gate_logits.detach().float(),
            }
            for gate, gate_logits in logits.items():
                post = 2 * torch.sigmoid(gate_logits)
                log_post = torch.log(post.clamp_min(self.log_eps))
                representations = {
                    # CLR removes a token-specific multiplicative common mode.
                    "clr_post_sigmoid": (
                        log_post - log_post.mean(dim=-1, keepdim=True)
                    ),
                    # First half of two-way centering. Dataset-wide dimension
                    # means are removed exactly when covariance is formed.
                    "centered_logits": (
                        gate_logits - gate_logits.mean(dim=-1, keepdim=True)
                    ),
                }
                for representation, value in representations.items():
                    for group, mask in self.token_masks.items():
                        self.values[(layer, gate, representation, group)].append(
                            value[mask].half()
                        )
        return self.original(
            activations, output_grad, down_weight,
            activation_gate_logits, gradient_gate_logits, eps,
        )


def centered_statistics(chunks, dimension, matmul_chunk=32768):
    cross = torch.zeros(dimension, dimension, device="cuda", dtype=torch.float64)
    total = torch.zeros(dimension, device="cuda", dtype=torch.float64)
    count = 0
    values = torch.cat(chunks, dim=0)
    for start in range(0, values.shape[0], matmul_chunk):
        x = values[start:start + matmul_chunk].float()
        cross.add_((x.T @ x).double())
        total.add_(x.sum(dim=0).double())
        count += x.shape[0]
    return cross.cpu(), total.cpu(), count


def residual_correlation(cross, total, count):
    covariance_sum = cross - torch.outer(total, total) / count
    variance_sum = torch.diag(covariance_sum).clamp_min(1e-30)
    denominator = torch.sqrt(variance_sum[:, None] * variance_sum[None, :])
    matrix = (covariance_sum / denominator).numpy()
    np.fill_diagonal(matrix, 1.0)
    return np.clip(matrix, -1.0, 1.0)


def clustered_order(matrix):
    distance = np.clip(1.0 - matrix, 0.0, 2.0)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    return leaves_list(linkage(
        squareform(distance, checks=False), method="average"
    )).astype(np.int32)


def plot_triplet(matrices, order, layer, gate, representation, counts, output):
    group_labels = {
        "assistant": "Assistant tokens",
        "non_assistant": "Non-assistant tokens",
        "all_valid": "All non-masked tokens",
    }
    representation_labels = {
        "clr_post_sigmoid": "Centered-log-ratio post-sigmoid gates",
        "centered_logits": "Two-way-centered pre-sigmoid logits",
    }
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    image = None
    for ax, group in zip(axes, GROUPS):
        matrix = matrices[group][np.ix_(order, order)]
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1,
                          interpolation="nearest", rasterized=True)
        ax.set_title(f"{group_labels[group]}\nN={counts[group]:,}")
        ax.set_xlabel("Clustered gate dimension")
        ax.set_ylabel("Clustered gate dimension")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=.84, label="Residual correlation")
    fig.suptitle(
        f"Layer {layer} {gate}-gate: {representation_labels[representation]}\n"
        "Ordering clustered from all-token residuals",
        fontsize=14,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    commit = os.popen(
        f"git -C {base.ORIGINAL_SRC.parent} rev-parse --short HEAD"
    ).read().strip()
    assert commit == "6d7726a", commit
    constants.CHECKPOINTS_PATH = args.checkpoint_root.resolve()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    model = load_checkpoint(
        "aklein4/Horizon-TPU_forte-1b", args.step, attention_kernel=None
    ).cuda().eval()
    model.lm_head.to(dtype=torch.bfloat16)
    enable_gradient_checkpointing(model, True)
    model.train()
    config = OmegaConf.load(base.ORIGINAL_SRC / "configs/data/horizons-llama3.yaml")
    stream = datasets.load_dataset(config.dataset.url, **config.dataset.kwargs)
    iterator = iter(stream)
    examples = [next(iterator) for _ in range(args.trajectories)]
    batch = HorizonCollator(**OmegaConf.to_container(
        config.collator.kwargs, resolve=True
    ))(examples)
    ids = batch["input_ids"].cuda()
    assistant = batch["assistant_mask"].cuda()
    attention = batch["attention_mask"].cuda()
    model.init_state(args.trajectories, torch.device("cuda"))

    collector = ResidualCollector(model)
    collector.install()
    try:
        for episode in range(64):
            loss = base.first_pass(
                model, ids[:, episode], assistant[:, episode],
                attention[:, episode], collector,
            )
            print(f"episode {episode:02d}: loss={loss:.6f}", flush=True)
    finally:
        collector.remove()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = {
        "trajectories": args.trajectories, "episodes": 64,
        "commit": commit, "checkpoint_step": args.step,
    }
    dimension = model.fast_modules()[0].fast_weight_size
    for layer in LAYERS:
        for gate in GATES:
            for representation in REPRESENTATIONS:
                cross, total, counts = {}, {}, {}
                for group in ("assistant", "non_assistant"):
                    cross[group], total[group], counts[group] = centered_statistics(
                        collector.values[(layer, gate, representation, group)],
                        dimension,
                    )
                cross["all_valid"] = cross["assistant"] + cross["non_assistant"]
                total["all_valid"] = total["assistant"] + total["non_assistant"]
                counts["all_valid"] = counts["assistant"] + counts["non_assistant"]
                matrices = {group: residual_correlation(
                    cross[group], total[group], counts[group]
                ) for group in GROUPS}
                order = clustered_order(matrices["all_valid"])
                prefix = f"layer{layer}_{gate}_gate_{representation}"
                for group in GROUPS:
                    archive[f"{prefix}_{group}"] = matrices[group]
                    archive[f"{prefix}_{group}_count"] = counts[group]
                archive[f"{prefix}_order"] = order
                plot_triplet(
                    matrices, order, layer, gate, representation, counts,
                    args.output_dir / f"{prefix}.png",
                )
                print(f"completed {prefix}", flush=True)
    np.savez_compressed(args.output_dir / "gate_residual_correlations.npz", **archive)
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
