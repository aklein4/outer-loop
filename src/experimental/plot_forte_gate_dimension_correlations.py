"""Cross-correlate Forte gate dimensions at layers 7 and 14.

Runs the original step-100 implementation from commit 6d7726a.  Correlation is
uncentered, exactly E[x_i x_j] / sqrt(E[x_i^2] E[x_j^2]).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ORIGINAL_SRC = Path(os.environ.get(
    "FORTE_ORIGINAL_SRC", "/tmp/forte-original-6d7726a/src"
)).resolve()
sys.path.insert(0, str(ORIGINAL_SRC))

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from collators.horizon import HorizonCollator
from models import load_checkpoint
import models.forte as forte_module
from models.forte import ForteMode
from utils import constants
from utils.torch_modules import enable_gradient_checkpointing


LAYERS = (7, 14)
GATES = ("a", "g")
GROUPS = ("assistant", "non_assistant", "all_valid")


def loss_and_lm_grad(model, states, input_ids, assistant_mask):
    labels = input_ids[:, 1:]
    weights = assistant_mask[:, 1:].float()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1) / states.shape[0]
    leaf = states.detach().requires_grad_(True)
    logits = model.lm_head(leaf).float()
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).reshape_as(labels)
    loss = (loss * weights).sum()
    loss.backward()
    return loss.detach(), leaf.grad.detach().to(states.dtype)


class GateCollector:
    def __init__(self, model, token_rms_norm=False, rms_norm_eps=1e-5):
        self.original = forte_module._get_G
        self.layer_by_weight = {
            module.down_fast.weight.data_ptr(): layer
            for layer, module in enumerate(model.fast_modules())
        }
        self.token_masks = None
        self.token_rms_norm = token_rms_norm
        self.rms_norm_eps = rms_norm_eps
        self.values = {
            (layer, gate, group): []
            for layer in LAYERS for gate in GATES
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
            gates = {
                "a": 2 * torch.sigmoid(activation_gate_logits.detach().float()),
                "g": 2 * torch.sigmoid(gradient_gate_logits.detach().float()),
            }
            for gate, value in gates.items():
                if self.token_rms_norm:
                    value = value * torch.rsqrt(
                        value.square().mean(dim=-1, keepdim=True)
                        + self.rms_norm_eps
                    )
                for group, mask in self.token_masks.items():
                    # FP16 storage is sufficient; second moments are accumulated
                    # in FP32 below. Keep tensors on GPU to avoid repeated copies.
                    self.values[(layer, gate, group)].append(value[mask].half())
        return self.original(
            activations, output_grad, down_weight,
            activation_gate_logits, gradient_gate_logits, eps,
        )


def first_pass(model, ids, assistant, attention, collector):
    collector.set_masks(assistant, attention)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, attention)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_backbone(
            ids, mode=ForteMode.TRAIN_FIRST,
            embeddings=embeddings, embedding_mask=attention,
        )
        states = model.forward_lm_states(
            hidden, mode=ForteMode.TRAIN_FIRST, logits_to_keep=slice(0, -1),
            embeddings=embeddings, embedding_mask=attention,
        )
        loss, grad = loss_and_lm_grad(model, states, ids, assistant)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(embeddings, attention, ForteMode.TRAIN_FIRST)
    return loss.item()


def second_moments(chunks, dimension, matmul_chunk=32768):
    cross = torch.zeros(dimension, dimension, device="cuda", dtype=torch.float64)
    count = 0
    # Concatenating episode chunks permits large, efficient GEMMs while bounding
    # temporary FP32 storage.
    values = torch.cat(chunks, dim=0)
    for start in range(0, values.shape[0], matmul_chunk):
        x = values[start:start + matmul_chunk].float()
        cross.add_((x.T @ x).double())
        count += x.shape[0]
    return cross.cpu(), count


def correlation(cross):
    diagonal = torch.diag(cross).clamp_min(1e-30)
    denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :])
    return (cross / denominator).numpy()


def clustered_order(matrix):
    distance = np.clip(1.0 - matrix, 0.0, 2.0)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    return leaves_list(tree).astype(np.int32)


def plot_triplet(matrices, order, layer, gate, counts, output):
    labels = {
        "assistant": "Assistant tokens",
        "non_assistant": "Non-assistant tokens",
        "all_valid": "All non-masked tokens",
    }
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    image = None
    for ax, group in zip(axes, GROUPS):
        sorted_matrix = matrices[group][np.ix_(order, order)]
        image = ax.imshow(sorted_matrix, cmap="cividis", vmin=0, vmax=1,
                          interpolation="nearest", rasterized=True)
        ax.set_title(f"{labels[group]}\nN={counts[group]:,}")
        ax.set_xlabel("Clustered gate dimension")
        ax.set_ylabel("Clustered gate dimension")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=.84, label="Uncentered correlation")
    fig.suptitle(
        f"Layer {layer} {gate}-gate dimension correlations\n"
        "Per-token RMS-normalized gates; ordering clustered from all tokens",
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
    parser.add_argument("--token-rms-norm", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available()
    commit = os.popen(
        f"git -C {ORIGINAL_SRC.parent} rev-parse --short HEAD"
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

    config = OmegaConf.load(ORIGINAL_SRC / "configs/data/horizons-llama3.yaml")
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

    collector = GateCollector(
        model, token_rms_norm=args.token_rms_norm,
        rms_norm_eps=model.config.rms_norm_eps,
    )
    collector.install()
    try:
        for episode in range(64):
            loss = first_pass(
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
        "token_rms_norm": args.token_rms_norm,
        "rms_norm_eps": model.config.rms_norm_eps,
    }
    for layer in LAYERS:
        for gate in GATES:
            cross = {}
            counts = {}
            for group in ("assistant", "non_assistant"):
                cross[group], counts[group] = second_moments(
                    collector.values[(layer, gate, group)],
                    model.fast_modules()[layer].fast_weight_size,
                )
            cross["all_valid"] = cross["assistant"] + cross["non_assistant"]
            counts["all_valid"] = counts["assistant"] + counts["non_assistant"]
            matrices = {group: correlation(cross[group]) for group in GROUPS}
            order = clustered_order(matrices["all_valid"])
            prefix = f"layer{layer}_{gate}_gate"
            for group in GROUPS:
                archive[f"{prefix}_{group}"] = matrices[group]
                archive[f"{prefix}_{group}_count"] = counts[group]
                np.savetxt(
                    args.output_dir / f"{prefix}_{group}.csv",
                    matrices[group], delimiter=",", fmt="%.8f",
                )
            archive[f"{prefix}_order"] = order
            np.savetxt(args.output_dir / f"{prefix}_order.csv", order,
                       delimiter=",", fmt="%d")
            plot_triplet(
                matrices, order, layer, gate, counts,
                args.output_dir / f"{prefix}_correlations.png",
            )
            print(f"completed {prefix}; counts={counts}", flush=True)
    np.savez_compressed(args.output_dir / "gate_dimension_correlations.npz", **archive)
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
