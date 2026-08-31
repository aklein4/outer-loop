"""Gate/offset ablations and log-LR analysis for the Forte-offset checkpoint.

This is deliberately CUDA-only and does not import or require torch-xla.
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


SCENARIOS = (
    "baseline",
    "gradient_token_mean",
    "gradient_batch_mean",
    "offset_token_mean",
    "offset_batch_mean",
    "both_token_mean",
    "both_batch_mean",
    "no_offset_update",
)
GATE_KINDS = ("gradient", "offset")
ROLES = ("all", "assistant", "nonassistant")


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x.dot(y) / den) if den else float("nan")


def _summary(x: np.ndarray) -> dict:
    q = np.quantile(x, [0, .01, .05, .5, .95, .99, 1])
    return {
        "mean": float(x.mean()), "std": float(x.std()), "rms": float(np.sqrt(np.mean(x*x))),
        "min": float(q[0]), "p01": float(q[1]), "p05": float(q[2]),
        "median": float(q[3]), "p95": float(q[4]), "p99": float(q[5]), "max": float(q[6]),
    }


class GateMoments:
    """Exact first/second moments across valid token gate vectors."""

    def __init__(self, layers: int, heads: int, device: torch.device):
        self.layers, self.heads = layers, heads
        self.n = torch.zeros(layers, len(ROLES), device=device, dtype=torch.float64)
        self.s = torch.zeros(layers, len(ROLES), 2, heads, device=device, dtype=torch.float64)
        self.cross = torch.zeros(layers, len(ROLES), 2, 2, heads, heads,
                                 device=device, dtype=torch.float64)

    @torch.no_grad()
    def add(self, layer: int, gradient: torch.Tensor, offset: torch.Tensor,
            valid: torch.Tensor, assistant: torch.Tensor) -> None:
        valid = valid.bool()
        assistant = assistant.bool()
        masks = (valid, valid & assistant, valid & ~assistant)
        gates = (gradient, offset)
        for role_i, mask in enumerate(masks):
            if not mask.any():
                continue
            xs = [g[mask].double() for g in gates]
            self.n[layer, role_i] += xs[0].shape[0]
            for a in range(2):
                self.s[layer, role_i, a] += xs[a].sum(0)
                for b in range(2):
                    self.cross[layer, role_i, a, b] += xs[a].T @ xs[b]

    def finalize(self) -> tuple[list[dict], dict[str, np.ndarray]]:
        rows, matrices = [], {}
        for layer in range(self.layers):
            for role_i, role in enumerate(ROLES):
                n = self.n[layer, role_i].item()
                means = self.s[layer, role_i] / max(n, 1)
                covs = {}
                for a in range(2):
                    for b in range(2):
                        covs[a, b] = self.cross[layer, role_i, a, b] / max(n, 1) - means[a, :, None] * means[b, None, :]
                for a, kind_a in enumerate(GATE_KINDS):
                    var_a = covs[a, a].diagonal().clamp_min(1e-30)
                    corr_aa = covs[a, a] / torch.sqrt(var_a[:, None] * var_a[None, :])
                    arr = corr_aa.float().cpu().numpy()
                    off = arr[~np.eye(self.heads, dtype=bool)]
                    eig = torch.linalg.eigvalsh(covs[a, a]).clamp_min(0)
                    pr = (eig.sum().square() / eig.square().sum().clamp_min(1e-30)).item()
                    rows.append({
                        "layer": layer, "role": role, "matrix": f"{kind_a}_{kind_a}", "tokens": int(n),
                        "mean_correlation": float(off.mean()), "mean_abs_correlation": float(np.abs(off).mean()),
                        "fraction_abs_gt_0.2": float((np.abs(off) > .2).mean()),
                        "fraction_abs_gt_0.5": float((np.abs(off) > .5).mean()),
                        "effective_dimensions": pr,
                    })
                    matrices[f"layer{layer}_{role}_{kind_a}_{kind_a}"] = arr
                var_g = covs[0, 0].diagonal().clamp_min(1e-30)
                var_o = covs[1, 1].diagonal().clamp_min(1e-30)
                corr_go = covs[0, 1] / torch.sqrt(var_g[:, None] * var_o[None, :])
                arr = corr_go.float().cpu().numpy()
                diagonal = np.diag(arr)
                rows.append({
                    "layer": layer, "role": role, "matrix": "gradient_offset", "tokens": int(n),
                    "mean_correlation": float(arr.mean()), "mean_abs_correlation": float(np.abs(arr).mean()),
                    "fraction_abs_gt_0.2": float((np.abs(arr) > .2).mean()),
                    "fraction_abs_gt_0.5": float((np.abs(arr) > .5).mean()),
                    "matched_head_mean_correlation": float(diagonal.mean()),
                    "matched_head_mean_abs_correlation": float(np.abs(diagonal).mean()),
                    "effective_dimensions": "",
                })
                matrices[f"layer{layer}_{role}_gradient_offset"] = arr
        return rows, matrices


class AblationRule:
    def __init__(self, model, logical_batch: int, moments: GateMoments):
        self.model, self.logical_batch, self.moments = model, logical_batch, moments
        self.original = forte._get_G
        self.module_by_weight = {m.down_fast.weight.data_ptr(): (i, m)
                                 for i, m in enumerate(model.fast_modules())}
        self.assistant: torch.Tensor | None = None

    def install(self):
        forte._get_G = self.get_g

    def remove(self):
        forte._get_G = self.original

    @staticmethod
    def transform(gate: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "none":
            return gate
        if mode == "token":
            return gate.mean(-1, keepdim=True).expand_as(gate)
        if mode == "batch":
            weights = mask.bool()[..., None].to(gate.dtype)
            mean = (gate * weights).sum((0, 1), keepdim=True) / weights.sum((0, 1), keepdim=True).clamp_min(1)
            return mean.expand_as(gate)
        raise ValueError(mode)

    def get_g(self, activations, output_grad, down_weight, valid_mask, *dynamic_args,
              eps: float, offset_alpha: float):
        layer, _ = self.module_by_weight[down_weight.data_ptr()]
        valid_mask = valid_mask.bool()
        offset, lr, offset_lr, gradient_logits, offset_logits = [x.float() for x in dynamic_args]
        mask = valid_mask[..., None].float()
        count = mask.sum(-2, keepdim=True).clamp_min(1)
        activations = activations.float() * mask
        g = F.linear(output_grad.float(), down_weight.float().T) * mask
        raw_G = torch.einsum("blo,bli->boi", g, activations)
        a_norm = activations * torch.rsqrt(activations.square().sum(-2, keepdim=True) / count + eps**2)
        g_norm = g * torch.rsqrt(g.square().sum(-2, keepdim=True) / count + eps**2)
        gradient_gate = unit_softplus(gradient_logits)
        offset_gate = unit_softplus(offset_logits)

        b = self.logical_batch
        if self.assistant is not None:
            self.moments.add(layer, gradient_gate[:b], offset_gate[:b],
                             valid_mask[:b], self.assistant[:b])

        updates = []
        for scenario_i, scenario in enumerate(SCENARIOS):
            sl = slice(scenario_i*b, (scenario_i+1)*b)
            gm = "token" if scenario in ("gradient_token_mean", "both_token_mean") else (
                 "batch" if scenario in ("gradient_batch_mean", "both_batch_mean") else "none")
            om = "token" if scenario in ("offset_token_mean", "both_token_mean") else (
                 "batch" if scenario in ("offset_batch_mean", "both_batch_mean") else "none")
            gg = self.transform(gradient_gate[sl], valid_mask[sl], gm)
            og = self.transform(offset_gate[sl], valid_mask[sl], om)
            base = torch.einsum("blo,bli->boi", forte.gate_heads(g_norm[sl], gg), a_norm[sl])
            base = -base * lr
            if scenario == "no_offset_update":
                updates.append(base)
                continue
            off = torch.einsum("blo,bli->boi", forte.gate_heads(offset[sl], og), a_norm[sl])
            off = -off * offset_lr
            base_norm = torch.norm(base, dim=(-2, -1), keepdim=True)
            off_norm = torch.norm(off, dim=(-2, -1), keepdim=True)
            updates.append(base + off * offset_alpha * base_norm / (off_norm + eps))
        return raw_G, torch.cat(updates, 0)


def loss_and_grad(model, states, ids, assistant, valid, logical_batch: int,
                  iterations: int, aux_weight: float):
    labels = ids[:, 1:]
    am, vm = assistant[:, 1:].float(), valid[:, 1:].float()
    nm = vm - am
    aw = am / am.sum(-1, keepdim=True).clamp_min(1)
    nw = nm / nm.sum(-1, keepdim=True).clamp_min(1)
    combined = aw + aux_weight * nw
    leaf = states.detach().reshape(-1, iterations, states.shape[-1]).requires_grad_(True)
    labels_i = labels.reshape(-1, iterations)
    weights_i = combined.reshape(-1, iterations)
    raw_parts = []
    for i in range(iterations):
        logits = model.lm_head(leaf[:, i]).float()
        raw = F.cross_entropy(logits, labels_i[:, i], reduction="none")
        (raw * weights_i[:, i]).sum().div(logical_batch).backward()
        raw_parts.append(raw.detach())
    raw_flat = torch.stack(raw_parts, 1).reshape_as(labels)
    assistant_loss = (raw_flat * aw).sum(-1)
    nonassistant_loss = (raw_flat * nw).sum(-1)
    return assistant_loss, nonassistant_loss, leaf.grad.reshape_as(states).detach().to(states.dtype)


def recurrent_step(model, ids, assistant, valid, logical_batch, iterations, aux_weight):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, valid).detach()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_backbone(ids, mode=ForteMode.TRAIN_FIRST,
                                        embeddings=embeddings, embedding_mask=valid)
        states = model.forward_lm_states(hidden, mode=ForteMode.TRAIN_FIRST,
                                        logits_to_keep=slice(0, -1), embeddings=embeddings,
                                        embedding_mask=valid)
        al, nl, grad = loss_and_grad(model, states, ids, assistant, valid,
                                     logical_batch, iterations, aux_weight)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(ForteMode.TRAIN_FIRST)
    return al.cpu().numpy(), nl.cpu().numpy()


def matrix_analysis(model, output_dir: Path) -> list[dict]:
    rows = []
    matrices = []
    for layer, module in enumerate(model.fast_modules()):
        base = (module.fast_dynamic_lr.log_lr.detach().float() * module.fast_dynamic_lr.scalar_scaler).cpu().numpy()
        offset = (module.fast_dynamic_lr.offset_log_lr.detach().float() * module.fast_dynamic_lr.scalar_scaler).cpu().numpy()
        matrices.append((base, offset))
        for kind, x, other in (("base", base, offset), ("offset", offset, base)):
            centered = x - x.mean()
            row_effect = x.mean(1, keepdims=True) - x.mean()
            col_effect = x.mean(0, keepdims=True) - x.mean()
            additive = row_effect + col_effect
            total_ss = float(np.square(centered).sum())
            fft_power = np.abs(np.fft.fftshift(np.fft.fft2(centered)))**2
            n = x.shape[0]
            yy, xx = np.ogrid[:n, :n]
            rr = np.sqrt((yy-(n-1)/2)**2 + (xx-(n-1)/2)**2)
            low = float(fft_power[rr <= n/16].sum() / fft_power.sum())
            # Singular values on GPU are substantially faster for these 1024x1024 matrices.
            s = torch.linalg.svdvals(torch.from_numpy(centered).cuda()).cpu().numpy()
            energy = s*s
            rows.append({
                "layer": layer, "matrix": kind, **_summary(x),
                "actual_lr_geomean": float(math.exp(x.mean()) * module.fast_dynamic_lr.base_lr / module.fast_weight_size),
                "actual_lr_arithmetic_mean": float((np.exp(x) * module.fast_dynamic_lr.base_lr / module.fast_weight_size).mean()),
                "corr_with_other_matrix": _corr(x, other),
                "corr_with_transpose": _corr(x, x.T),
                "horizontal_neighbor_corr": _corr(x[:, :-1], x[:, 1:]),
                "vertical_neighbor_corr": _corr(x[:-1, :], x[1:, :]),
                "row_col_additive_variance_fraction": float(np.square(additive).sum() / total_ss),
                "low_frequency_power_fraction": low,
                "sv_top1_energy_fraction": float(energy[0] / energy.sum()),
                "sv_top8_energy_fraction": float(energy[:8].sum() / energy.sum()),
                "sv_top32_energy_fraction": float(energy[:32].sum() / energy.sum()),
                "sv_effective_rank": float(energy.sum()**2 / np.square(energy).sum()),
            })

    fig, axes = plt.subplots(len(matrices), 3, figsize=(12, 3*len(matrices)), constrained_layout=True)
    for layer, (base, offset) in enumerate(matrices):
        triplet = (base, offset, offset-base)
        titles = ("base Δlog LR", "offset Δlog LR", "offset − base")
        for j, (x, title) in enumerate(zip(triplet, titles)):
            lim = np.quantile(np.abs(x - (0 if j == 2 else x.mean())), .995)
            center = 0 if j == 2 else x.mean()
            im = axes[layer, j].imshow(x, cmap="coolwarm", vmin=center-lim, vmax=center+lim,
                                       interpolation="nearest", rasterized=True)
            axes[layer, j].set_title(f"L{layer} {title}")
            axes[layer, j].set_xticks([]); axes[layer, j].set_yticks([])
            fig.colorbar(im, ax=axes[layer, j], fraction=.035, pad=.02)
    fig.savefig(output_dir / "log_lr_matrices_all_layers.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for layer, (base, offset) in enumerate(matrices):
        axes[0, 0].plot(base.mean(1), alpha=.55, lw=.7)
        axes[0, 1].plot(offset.mean(1), alpha=.55, lw=.7)
        axes[1, 0].plot(base.mean(0), alpha=.55, lw=.7)
        axes[1, 1].plot(offset.mean(0), alpha=.55, lw=.7)
    for ax, title in zip(axes.flat, ("Base row means", "Offset row means", "Base column means", "Offset column means")):
        ax.set_title(title); ax.set_xlabel("matrix coordinate"); ax.set_ylabel("mean Δlog LR")
    fig.savefig(output_dir / "log_lr_row_column_profiles.png", dpi=180)
    plt.close(fig)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def paired_summary(loss_rows: list[dict]) -> list[dict]:
    # Aggregate chunks 1..end per trajectory, pair each intervention with its baseline.
    by = defaultdict(list)
    for r in loss_rows:
        if r["chunk"] > 0:
            by[(r["trajectory"], r["scenario"], "assistant")].append(r["assistant_loss"])
            by[(r["trajectory"], r["scenario"], "nonassistant")].append(r["nonassistant_loss"])
            by[(r["trajectory"], r["scenario"], "weighted_total")].append(r["weighted_total_loss"])
    means = {k: np.mean(v) for k, v in by.items()}
    trajectories = sorted({r["trajectory"] for r in loss_rows})
    out = []
    for scenario in SCENARIOS[1:]:
        for objective in ("assistant", "nonassistant", "weighted_total"):
            base = np.array([means[t, "baseline", objective] for t in trajectories])
            abl = np.array([means[t, scenario, objective] for t in trajectories])
            delta = abl-base
            se = delta.std(ddof=1)/math.sqrt(len(delta))
            out.append({
                "scenario": scenario, "objective": objective, "trajectories": len(delta),
                "baseline_loss": float(base.mean()), "ablated_loss": float(abl.mean()),
                "delta_loss": float(delta.mean()), "delta_percent": float(100*delta.mean()/base.mean()),
                "ci95_low": float(delta.mean()-1.96*se), "ci95_high": float(delta.mean()+1.96*se),
                "positive_trajectories": int((delta > 0).sum()),
            })
    return out


def plot_ablation(summary: list[dict], output: Path) -> None:
    rows = [r for r in summary if r["objective"] in ("assistant", "nonassistant")]
    scenarios = list(SCENARIOS[1:])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, objective in zip(axes, ("assistant", "nonassistant")):
        rs = {r["scenario"]: r for r in rows if r["objective"] == objective}
        y = np.array([rs[s]["delta_percent"] for s in scenarios])
        base = np.array([rs[s]["baseline_loss"] for s in scenarios])
        lo = np.array([100*rs[s]["ci95_low"]/base[i] for i, s in enumerate(scenarios)])
        hi = np.array([100*rs[s]["ci95_high"]/base[i] for i, s in enumerate(scenarios)])
        ax.bar(np.arange(len(scenarios)), y, color="#4472c4")
        ax.errorbar(np.arange(len(scenarios)), y, yerr=[y-lo, hi-y], fmt="none", color="black", capsize=3)
        ax.axhline(0, color="black", lw=.8); ax.set_title(f"{objective.title()} loss")
        ax.set_ylabel("paired Δloss (%)"); ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], rotation=25, ha="right", fontsize=8)
    fig.savefig(output, dpi=180); plt.close(fig)


def plot_gate_correlations(matrices: dict[str, np.ndarray], output: Path, layers: int) -> None:
    fig, axes = plt.subplots(layers, 3, figsize=(9, 3*layers), constrained_layout=True)
    names = ("gradient_gradient", "offset_offset", "gradient_offset")
    for layer in range(layers):
        for j, name in enumerate(names):
            arr = matrices[f"layer{layer}_all_{name}"]
            im = axes[layer, j].imshow(arr, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
            axes[layer, j].set_title(f"L{layer} {name.replace('_', '↔', 1)}")
            axes[layer, j].set_xticks([]); axes[layer, j].set_yticks([])
    fig.colorbar(im, ax=axes, shrink=.25, label="Pearson correlation")
    fig.savefig(output, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="aklein4/horizon-v2_forte-offset")
    ap.add_argument("--step", type=int, default=150)
    ap.add_argument("--trajectories", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=64)
    ap.add_argument("--data-config", type=Path, default=SRC/"configs/data/300k-horizons-llama3.yaml")
    ap.add_argument("--output-dir", type=Path, default=SRC/"local_data/horizon_v2_forte_offset_step150_analysis")
    ap.add_argument("--aux-loss-weight", type=float, default=.1)
    ap.add_argument("--num-logit-iterations", type=int, default=4)
    ap.add_argument("--skip-matrix-analysis", action="store_true")
    args = ap.parse_args()
    if args.trajectories % args.batch_size:
        raise ValueError("trajectories must be divisible by batch size")
    assert torch.cuda.is_available()
    torch.manual_seed(42); np.random.seed(42); torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(args.checkpoint, args.step, attention_kernel=None).cuda().eval()
    model.lm_head.to(torch.bfloat16)
    enable_gradient_checkpointing(model, True); model.train()
    if not args.skip_matrix_analysis:
        matrix_rows = matrix_analysis(model, args.output_dir)
        write_csv(args.output_dir/"log_lr_matrix_metrics.csv", matrix_rows)

    cfg = OmegaConf.load(args.data_config)
    stream = datasets.load_dataset(cfg.dataset.url, **cfg.dataset.kwargs)
    iterator = iter(stream)
    raw = [next(iterator) for _ in range(args.trajectories)]
    collator_kwargs = OmegaConf.to_container(cfg.collator.kwargs, resolve=True)
    collator_kwargs["cluster_length"] = args.chunks
    collator = HorizonCollator(**collator_kwargs)
    layers = len(list(model.fast_modules()))
    moments = GateMoments(layers, model.config.num_fast_weight_heads, torch.device("cuda"))
    loss_rows = []
    trajectory_base = 0
    for start in range(0, args.trajectories, args.batch_size):
        batch = collator(raw[start:start+args.batch_size])
        base_ids, base_assistant, base_valid = (batch[k].cuda() for k in ("input_ids", "assistant_mask", "attention_mask"))
        ids = torch.cat([base_ids for _ in SCENARIOS], 0)
        assistant = torch.cat([base_assistant for _ in SCENARIOS], 0)
        valid = torch.cat([base_valid for _ in SCENARIOS], 0)
        physical_batch = args.batch_size * len(SCENARIOS)
        model.init_state(physical_batch, torch.device("cuda"))
        rule = AblationRule(model, args.batch_size, moments); rule.install()
        try:
            for chunk in range(args.chunks):
                rule.assistant = assistant[:, chunk]
                al, nl = recurrent_step(model, ids[:, chunk], assistant[:, chunk], valid[:, chunk],
                                        args.batch_size, args.num_logit_iterations, args.aux_loss_weight)
                for scenario_i, scenario in enumerate(SCENARIOS):
                    for local_t in range(args.batch_size):
                        idx = scenario_i*args.batch_size+local_t
                        loss_rows.append({
                            "trajectory": start+local_t, "batch": start//args.batch_size,
                            "chunk": chunk, "scenario": scenario,
                            "assistant_loss": float(al[idx]), "nonassistant_loss": float(nl[idx]),
                            "weighted_total_loss": float(al[idx]+args.aux_loss_weight*nl[idx]),
                            "valid_tokens": int(base_valid[local_t, chunk].sum()),
                            "assistant_tokens": int(base_assistant[local_t, chunk].sum()),
                        })
                print(f"batch {start//args.batch_size+1}/{args.trajectories//args.batch_size} chunk {chunk+1}/{args.chunks} "
                      f"baseline assistant={al[:args.batch_size].mean():.5f}", flush=True)
        finally:
            rule.remove()
        del ids, assistant, valid, base_ids, base_assistant, base_valid, batch
        trajectory_base += args.batch_size

    write_csv(args.output_dir/"ablation_losses.csv", loss_rows)
    ablation_summary = paired_summary(loss_rows)
    write_csv(args.output_dir/"ablation_summary.csv", ablation_summary)
    plot_ablation(ablation_summary, args.output_dir/"ablation_summary.png")
    gate_rows, gate_matrices = moments.finalize()
    write_csv(args.output_dir/"gate_correlation_summary.csv", gate_rows)
    np.savez_compressed(args.output_dir/"gate_correlation_matrices.npz", **gate_matrices)
    plot_gate_correlations(gate_matrices, args.output_dir/"gate_correlation_heatmaps.png", layers)
    metadata = {
        "checkpoint": args.checkpoint, "step": args.step, "data_config": str(args.data_config),
        "dataset": cfg.dataset.url, "trajectories": args.trajectories, "batch_size": args.batch_size,
        "chunks": args.chunks, "max_length": cfg.collator.kwargs.max_length,
        "scenarios": list(SCENARIOS), "gate_units": "post-unit_softplus",
        "token_mean_definition": "mean over 32 heads independently for each token, broadcast to all heads",
        "batch_mean_definition": "valid-token mean independently for each head within each minibatch and chunk, broadcast to all tokens",
        "no_offset_definition": "remove the complete alpha-scaled offset update after computing the base update",
        "loss_chunks_summarized": "1..63 (chunk 0 is pre-update and identical)",
        "aux_loss_weight": args.aux_loss_weight, "num_logit_iterations": args.num_logit_iterations,
        "gradient_checkpointing": True, "torch_xla_used": False,
        "cuda_device": torch.cuda.get_device_name(), "peak_cuda_gb": torch.cuda.max_memory_allocated()/2**30,
    }
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "peak_cuda_gb": metadata["peak_cuda_gb"]}, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
