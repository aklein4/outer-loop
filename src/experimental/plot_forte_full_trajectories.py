"""Run complete Forte trajectories and plot gate/update diagnostics on CUDA."""

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

from collators.horizon import HorizonCollator
from models import load_checkpoint
import models.forte as forte_module
from models.forte import ForteMode
from experimental.diagnose_forte_gates import loss_and_lm_grad
from utils.torch_modules import enable_gradient_checkpointing


def cosine_per_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().flatten(1)
    y = y.float().flatten(1)
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-30)


def norm_per_batch(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(1).norm(dim=1)


class TrajectoryCollector:
    def __init__(self, model: torch.nn.Module, masks: torch.Tensor):
        self.masks = masks
        self.position = -1
        self.rows: list[dict] = []
        self.layer_by_weight = {
            mlp.down_fast.weight.data_ptr(): i
            for i, mlp in enumerate(model.fast_modules())
        }
        self.original = forte_module._get_G

    def install(self) -> None:
        forte_module._get_G = self.get_g

    def remove(self) -> None:
        forte_module._get_G = self.original

    def get_g(self, activations, output_grad, down_weight,
              activation_gate_logits, gradient_gate_logits, valid_mask, eps):
        raw_g, update = self.original(
            activations, output_grad, down_weight,
            activation_gate_logits, gradient_gate_logits, valid_mask, eps,
        )
        layer = self.layer_by_weight[down_weight.data_ptr()]
        a = activations.float()
        g = torch.nn.functional.linear(output_grad.float(), down_weight.T.float())
        a_norm = a * torch.rsqrt(a.square().mean(dim=-2, keepdim=True) + eps)
        g_norm = g * torch.rsqrt(g.square().sum(dim=-2, keepdim=True) + eps)
        ag = 2 * torch.sigmoid(activation_gate_logits.float())
        gg = 2 * torch.sigmoid(gradient_gate_logits.float())
        normalized_g = g_norm.mT @ a_norm
        activation_only = g_norm.mT @ (ag * a_norm)
        gradient_only = (gg * g_norm).mT @ a_norm

        mask = self.masks[:, self.position].to(ag.device)
        mask_f = mask[..., None].float()
        valid_count = mask_f.sum(dim=-2, keepdim=True).clamp_min(1)
        g_valid = g * mask_f
        a_valid_mean = (
            a
            * torch.rsqrt(a.square().sum(dim=-2, keepdim=True) / valid_count + eps)
        )
        g_valid_rms = (
            g_valid
            * torch.rsqrt(g_valid.square().sum(dim=-2, keepdim=True) / valid_count + eps)
        )
        mask_mean_a_only = g_norm.mT @ a_valid_mean
        mask_mean_g_only = g_valid_rms.mT @ a_norm
        mask_mean_both = g_valid_rms.mT @ a_valid_mean
        # Proposed variant: both the mean and the explicit square-root scale
        # count valid positions only. State updates still use `update` above.
        g_valid_mean = (
            g_valid
            * torch.rsqrt(g_valid.square().sum(dim=-2, keepdim=True) / valid_count + eps)
            * torch.sqrt(valid_count)
        )
        valid_mean_ungated = g_valid_mean.mT @ a_norm
        valid_mean_update = (gg * g_valid_mean).mT @ (ag * a_norm)
        total_g_energy = g.square().sum(dim=(-2, -1))
        valid_g_energy = g_valid.square().sum(dim=(-2, -1))
        for trajectory in range(ag.shape[0]):
            am = ag[trajectory][mask[trajectory]]
            gm = gg[trajectory][mask[trajectory]]
            al = activation_gate_logits.float()[trajectory][mask[trajectory]]
            gl = gradient_gate_logits.float()[trajectory][mask[trajectory]]
            effective = am * gm
            row = {
                "trajectory": trajectory,
                "position": self.position,
                "layer": layer,
                "tokens": int(mask[trajectory].sum()),
                "activation_gate_mean": am.mean().item(),
                "activation_gate_rms": am.square().mean().sqrt().item(),
                "gradient_gate_mean": gm.mean().item(),
                "gradient_gate_rms": gm.square().mean().sqrt().item(),
                "effective_gate_mean": effective.mean().item(),
                "effective_gate_geometric_mean": effective.clamp_min(1e-30).log().mean().exp().item(),
                "activation_logit_min": al.min().item(),
                "activation_logit_max": al.max().item(),
                "gradient_logit_min": gl.min().item(),
                "gradient_logit_max": gl.max().item(),
            }
            for name, gate in (("activation", am), ("gradient", gm)):
                for threshold in (.01, .05, .1):
                    row[f"{name}_gate_lt_{threshold:g}"] = (gate < threshold).float().mean().item()
                    row[f"{name}_gate_gt_{2-threshold:g}"] = (gate > 2-threshold).float().mean().item()
            for name, matrix in (
                ("normalized_G", normalized_g),
                ("activation_only", activation_only),
                ("gradient_only", gradient_only),
                ("update", update),
            ):
                row[f"cos_{name}_vs_raw_G"] = cosine_per_batch(matrix, raw_g)[trajectory].item()
                row[f"norm_{name}_over_raw_G"] = (
                    norm_per_batch(matrix)[trajectory]
                    / norm_per_batch(raw_g)[trajectory].clamp_min(1e-30)
                ).item()
            row["raw_G_norm"] = norm_per_batch(raw_g)[trajectory].item()
            row["g_energy_valid_fraction"] = (
                valid_g_energy[trajectory] / total_g_energy[trajectory].clamp_min(1e-30)
            ).item()
            row["norm_valid_mean_ungated_over_raw_G"] = (
                norm_per_batch(valid_mean_ungated)[trajectory]
                / norm_per_batch(raw_g)[trajectory].clamp_min(1e-30)
            ).item()
            row["norm_valid_mean_update_over_raw_G"] = (
                norm_per_batch(valid_mean_update)[trajectory]
                / norm_per_batch(raw_g)[trajectory].clamp_min(1e-30)
            ).item()
            row["cos_valid_mean_ungated_vs_raw_G"] = cosine_per_batch(
                valid_mean_ungated, raw_g
            )[trajectory].item()
            row["cos_valid_mean_update_vs_raw_G"] = cosine_per_batch(
                valid_mean_update, raw_g
            )[trajectory].item()
            row["norm_valid_mean_update_over_current_update"] = (
                norm_per_batch(valid_mean_update)[trajectory]
                / norm_per_batch(update)[trajectory].clamp_min(1e-30)
            ).item()
            for name, matrix in (
                ("mask_mean_a_only", mask_mean_a_only),
                ("mask_mean_g_only", mask_mean_g_only),
                ("mask_mean_both", mask_mean_both),
            ):
                row[f"norm_{name}_over_raw_G"] = (
                    norm_per_batch(matrix)[trajectory]
                    / norm_per_batch(raw_g)[trajectory].clamp_min(1e-30)
                ).item()
            self.rows.append(row)
        return raw_g, update


def first_pass(model, ids, assistant, mask, collector):
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
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(embeddings, mask, ForteMode.TRAIN_FIRST)
    return loss.item()


def mean_grid(rows: list[dict], key: str) -> np.ndarray:
    grid = np.full((16, 64), np.nan)
    for layer in range(16):
        for position in range(64):
            values = [r[key] for r in rows if r["layer"] == layer and r["position"] == position]
            grid[layer, position] = np.mean(values)
    return grid


def plot_lines(rows, key, ylabel, title, path):
    grid = mean_grid(rows, key)
    fig, ax = plt.subplots(figsize=(11, 6))
    for layer in range(16):
        ax.plot(range(64), grid[layer], label=f"L{layer}", linewidth=1.4,
                alpha=.9 if layer in (0, 1, 2, 6, 10, 14, 15) else .45)
    ax.axhline(1, color="black", linestyle="--", linewidth=1, alpha=.5)
    ax.set(xlabel="Episode position", ylabel=ylabel, title=title, xlim=(0, 63))
    ax.grid(alpha=.2)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(rows, key, title, path, vmin=None, vmax=None, cmap="viridis"):
    grid = mean_grid(rows, key)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest",
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set(xlabel="Episode position", ylabel="Layer", title=title)
    ax.set_yticks(range(16))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="aklein4/Horizon-TPU_forte-1b")
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--trajectories", type=int, default=3)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("forte_full_trajectories_step50"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    model = load_checkpoint(args.repo, args.step, attention_kernel=None).cuda().eval()
    model.lm_head.to(dtype=torch.bfloat16)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model, True)
        # LayerStack checkpointing follows PyTorch's standard training-mode guard.
        # This model has no active dropout, so training mode does not change its math.
        model.train()
    cfg = OmegaConf.load(SRC / "configs/data/horizons-llama3.yaml")
    stream = datasets.load_dataset(cfg.dataset.url, **cfg.dataset.kwargs)
    iterator = iter(stream)
    raw = [next(iterator) for _ in range(args.trajectories)]
    batch = HorizonCollator(**OmegaConf.to_container(cfg.collator.kwargs, resolve=True))(raw)
    ids = batch["input_ids"].cuda()
    assistant = batch["assistant_mask"].cuda()
    masks = batch["attention_mask"].cuda()
    model.init_state(args.trajectories, torch.device("cuda"))
    collector = TrajectoryCollector(model, masks)
    collector.install()
    losses = []
    state_norms = []
    try:
        for position in range(64):
            collector.position = position
            losses.append(first_pass(
                model, ids[:, position], assistant[:, position], masks[:, position], collector,
            ))
            state_norms.append([
                mlp.state.float().flatten(1).norm(dim=1).cpu().tolist()
                for mlp in model.fast_modules()
            ])
            print(f"position {position:02d}: loss={losses[-1]:.6f}", flush=True)
    finally:
        collector.remove()

    rows = collector.rows
    with (args.output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lr_rows = []
    with torch.no_grad():
        for layer, mlp in enumerate(model.fast_modules()):
            lr = torch.exp(
                mlp.fast_dynamic_lr.log_lr.float() * math.sqrt(mlp.fast_weight_size)
                + math.log(mlp.fast_dynamic_lr.base_lr)
                - math.log(mlp.fast_weight_size)
            ).flatten()
            q = torch.quantile(lr, torch.tensor([0, .001, .01, .5, .99, .999, 1], device=lr.device))
            lr_rows.append({
                "layer": layer, "mean": lr.mean().item(), "std": lr.std(unbiased=False).item(),
                **{f"q_{name}": value for name, value in zip(
                    ("0", "001", "01", "50", "99", "999", "100"), q.cpu().tolist())},
            })

    metadata = {
        "checkpoint": {"repo": args.repo, "step": args.step},
        "trajectories": args.trajectories,
        "sources": [r.get("source") for r in raw],
        "cluster_num_tokens": [r.get("num_tokens") for r in raw],
        "nonpad_tokens": masks.sum(dim=(1, 2)).cpu().tolist(),
        "assistant_tokens": assistant.sum(dim=(1, 2)).cpu().tolist(),
        "losses": losses,
        "state_norms": state_norms,
        "dynamic_learning_rates": lr_rows,
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    plot_lines(rows, "activation_gate_mean", "Mean 2 sigmoid(logit)",
               "Activation gate magnitude across full trajectories",
               args.output_dir / "activation_gate_by_position.png")
    plot_lines(rows, "gradient_gate_mean", "Mean 2 sigmoid(logit)",
               "Gradient gate magnitude across full trajectories",
               args.output_dir / "gradient_gate_by_position.png")
    plot_heatmap(rows, "effective_gate_geometric_mean",
                 "Geometric mean of activation × gradient gate",
                 args.output_dir / "effective_gate_geomean_heatmap.png", vmin=0, vmax=1)
    plot_heatmap(rows, "cos_normalized_G_vs_raw_G",
                 "Cosine: normalized ungated G vs raw G",
                 args.output_dir / "cos_normalized_G_vs_raw_G.png", vmin=0, vmax=1, cmap="magma")
    plot_heatmap(rows, "cos_update_vs_raw_G",
                 "Cosine: fully gated update vs raw G",
                 args.output_dir / "cos_update_vs_raw_G.png", vmin=0, vmax=1, cmap="magma")
    plot_heatmap(rows, "activation_gate_lt_0.1",
                 "Fraction of activation gates below 0.1",
                 args.output_dir / "activation_saturation_low.png", vmin=0, vmax=.25, cmap="Reds")
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "rows": len(rows),
        "sources": metadata["sources"],
        "peak_cuda_gb": metadata["peak_cuda_gb"],
    }, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
