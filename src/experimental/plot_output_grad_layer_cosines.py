"""Average cross-layer cosine similarity of pre-down-projection Forte gradients.

This probe is intended to run against the original step-100 implementation at
commit 6d7726a, independently of the current working tree.
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

from collators.horizon import HorizonCollator
from models import load_checkpoint
import models.forte as forte_module
from models.forte import ForteMode
from utils import constants
from utils.torch_modules import enable_gradient_checkpointing


def loss_and_lm_grad(model, states, input_ids, assistant_mask):
    labels = input_ids[:, 1:]
    weights = assistant_mask[:, 1:].float()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1) / states.shape[0]
    leaf = states.detach().requires_grad_(True)
    logits = model.lm_head(leaf).float()
    losses = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).reshape_as(labels)
    loss = (losses * weights).sum()
    loss.backward()
    return loss.detach(), leaf.grad.detach().to(states.dtype)


class Collector:
    def __init__(self, model):
        self.original = forte_module._get_G
        self.layer_by_weight = {
            module.down_fast.weight.data_ptr(): layer
            for layer, module in enumerate(model.fast_modules())
        }
        self.output_grads = {}

    def install(self):
        forte_module._get_G = self.get_g

    def remove(self):
        forte_module._get_G = self.original

    def reset(self):
        self.output_grads.clear()

    def get_g(self, activations, output_grad, down_weight,
              activation_gate_logits, gradient_gate_logits, eps):
        layer = self.layer_by_weight[down_weight.data_ptr()]
        if layer in self.output_grads:
            raise RuntimeError(f"recorded layer {layer} more than once")
        # This is deliberately captured before fixed_linear(..., down_weight.T).
        self.output_grads[layer] = output_grad.detach().float()
        return self.original(
            activations, output_grad, down_weight,
            activation_gate_logits, gradient_gate_logits, eps,
        )


def accumulate_cosines(output_grads, masks, eps, sums, counts):
    """Normalize each feature over selected sequence positions, then tokens."""
    x = torch.stack([output_grads[i] for i in sorted(output_grads)], dim=2)
    # x: batch, sequence, layer, hidden
    for name, mask in masks.items():
        selected = mask.bool()
        selected4 = selected[:, :, None, None]
        n = selected.sum(1, keepdim=True).clamp_min(1).float()[:, :, None, None]
        mean_square = (x.square() * selected4).sum(dim=1, keepdim=True) / n
        sequence_rms_normalized = x * torch.rsqrt(mean_square + eps)
        unit = torch.nn.functional.normalize(
            sequence_rms_normalized, p=2, dim=-1, eps=1e-30
        )
        chosen = unit[selected]  # selected-token, layer, hidden
        sums[name] += torch.einsum("nld,nmd->lm", chosen, chosen).double().cpu()
        counts[name] += chosen.shape[0]


def first_pass(model, ids, assistant, mask, collector, sums, counts):
    collector.reset()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_backbone(
            ids, mode=ForteMode.TRAIN_FIRST,
            embeddings=embeddings, embedding_mask=mask,
        )
        states = model.forward_lm_states(
            hidden, mode=ForteMode.TRAIN_FIRST, logits_to_keep=slice(0, -1),
            embeddings=embeddings, embedding_mask=mask,
        )
        loss, grad = loss_and_lm_grad(model, states, ids, assistant)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    if len(collector.output_grads) != len(collector.layer_by_weight):
        raise RuntimeError(
            f"captured {len(collector.output_grads)} gradients for "
            f"{len(collector.layer_by_weight)} layers"
        )
    # The backbone gradients cover every hidden-token position (the shifted LM
    # objective propagates through the full causal sequence).  Compare matching
    # hidden positions and classify them by the token resident at that position.
    valid = mask.bool()
    assistant_token = assistant.bool() & valid
    masks = {
        "assistant": assistant_token,
        "non_assistant": valid & ~assistant_token,
        "all_valid": valid,
    }
    accumulate_cosines(
        collector.output_grads, masks, model.fast_modules()[0].grad_eps,
        sums, counts,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(embeddings, mask, ForteMode.TRAIN_FIRST)
    return loss.item()


def plot_matrices(matrices, counts, output):
    colors = plt.get_cmap("cividis")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
    titles = {
        "assistant": "Assistant tokens",
        "non_assistant": "Non-assistant tokens",
        "all_valid": "All non-masked tokens",
    }
    image = None
    for ax, (name, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=colors,
                          interpolation="nearest", origin="upper")
        ax.set_title(f"{titles[name]}\nN={counts[name]:,}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ticks = np.arange(matrix.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=axes, shrink=.84, label="Average cosine similarity")
    fig.suptitle(
        "Pre-down-projection output-gradient similarity across layers\n"
        "Mask-aware sequence RMS, then per-token L2 normalization",
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
    masks = batch["attention_mask"].cuda()
    model.init_state(args.trajectories, torch.device("cuda"))

    num_layers = len(model.fast_modules())
    names = ("assistant", "non_assistant", "all_valid")
    sums = {name: torch.zeros(num_layers, num_layers, dtype=torch.float64)
            for name in names}
    counts = {name: 0 for name in names}
    collector = Collector(model)
    collector.install()
    try:
        for position in range(64):
            loss = first_pass(
                model, ids[:, position], assistant[:, position],
                masks[:, position], collector, sums, counts,
            )
            print(f"episode {position:02d}: loss={loss:.6f}", flush=True)
    finally:
        collector.remove()

    matrices = {name: (sums[name] / counts[name]).numpy() for name in names}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "output_grad_layer_cosines_step100.npz",
        **matrices, **{f"{name}_count": count for name, count in counts.items()},
        trajectories=args.trajectories, episodes=64, commit=commit,
        checkpoint_step=args.step,
    )
    for name, matrix in matrices.items():
        np.savetxt(
            args.output_dir / f"output_grad_layer_cosines_{name}_step100.csv",
            matrix, delimiter=",", fmt="%.8f",
        )
    plot_matrices(
        matrices, counts,
        args.output_dir / "output_grad_layer_cosines_step100.png",
    )
    print(f"counts: {counts}")
    print(f"wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
