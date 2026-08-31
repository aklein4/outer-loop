"""Compare pre-spike and recovered Forte-offset checkpoints on real Horizons data.

CUDA/PyTorch only.  This file intentionally does not import torch-xla.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator  # noqa: E402
from models import _clean_wrapped_state_dict, load_checkpoint  # noqa: E402
import models.forte as forte  # noqa: E402
from models.forte import ForteMode  # noqa: E402
from utils.torch_modules import enable_gradient_checkpointing  # noqa: E402
from utils.torch_utils import unit_softplus  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rms(x: torch.Tensor) -> float:
    return x.detach().float().square().mean().sqrt().item()


def norm_rows(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().flatten(1).norm(dim=1)


def cosine_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xf, yf = x.detach().float().flatten(1), y.detach().float().flatten(1)
    return (xf * yf).sum(1) / (xf.norm(dim=1) * yf.norm(dim=1)).clamp_min(1e-30)


def masked_stats(x: torch.Tensor, mask: torch.Tensor) -> dict:
    values = x.detach().float()[mask.bool()]
    if not values.numel():
        return {k: float("nan") for k in ("mean", "std", "rms", "p01", "p50", "p99", "min", "max")}
    flat = values.reshape(-1)
    q = torch.quantile(flat, torch.tensor([.01, .5, .99], device=flat.device))
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "rms": flat.square().mean().sqrt().item(),
        "p01": q[0].item(), "p50": q[1].item(), "p99": q[2].item(),
        "min": flat.min().item(), "max": flat.max().item(),
    }


def tensor_family(name: str) -> str:
    for key in (
        "gradient_gate_proj", "offset_gate_proj", "offset_proj", "offset_log_lr",
        "log_lr", "up_fast", "gate_fast", "down_fast", "embedding_state",
        "bidirectional_head", "embed_tokens", "lm_head", "self_attn",
        "gate_proj", "up_proj", "down_proj", "lm_norm",
    ):
        if key in name:
            return key
    return "other"


def tensor_layer(name: str) -> int | None:
    m = re.search(r"(?:backbone|output)_layers\.layers\.(\d+)", name)
    if not m:
        return None
    layer = int(m.group(1))
    return layer if name.startswith("backbone") else layer + 12


def analyze_parameter_diffs(checkpoint_root: Path, steps: list[int], output_dir: Path) -> dict:
    print("Loading state dicts for parameter displacement analysis", flush=True)
    states = {}
    for step in steps:
        path = checkpoint_root / f"{step:012d}" / "model.pt"
        states[step] = _clean_wrapped_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    common = sorted(set.intersection(*(set(s) for s in states.values())))
    rows = []
    group_acc = defaultdict(lambda: defaultdict(float))
    s0, s1, s2 = steps
    for i, name in enumerate(common):
        a, b, c = (states[s][name].float() for s in steps)
        if not a.is_floating_point():
            continue
        pre, event = b - a, c - b
        n = a.numel()
        pre_ss, event_ss = pre.square().sum().item(), event.square().sum().item()
        p200_ss = b.square().sum().item()
        dot = (pre * event).sum().item()
        row = {
            "name": name, "family": tensor_family(name), "layer": tensor_layer(name), "elements": n,
            "step150_rms": math.sqrt(a.square().mean().item()),
            "step200_rms": math.sqrt(p200_ss / n),
            "step250_rms": math.sqrt(c.square().mean().item()),
            "delta_150_200_rms": math.sqrt(pre_ss / n),
            "delta_200_250_rms": math.sqrt(event_ss / n),
            "event_over_pre_delta_rms": math.sqrt(event_ss / max(pre_ss, 1e-60)),
            "event_delta_over_param_rms": math.sqrt(event_ss / max(p200_ss, 1e-60)),
            "pre_event_delta_cosine": dot / math.sqrt(max(pre_ss * event_ss, 1e-60)),
            "event_delta_max_abs": event.abs().max().item(),
            "step200_max_abs": b.abs().max().item(), "step250_max_abs": c.abs().max().item(),
            "finite_200": bool(torch.isfinite(b).all()), "finite_250": bool(torch.isfinite(c).all()),
        }
        rows.append(row)
        for group in ("all", row["family"], f"layer_{row['layer']}" if row["layer"] is not None else "noncausal"):
            acc = group_acc[group]
            acc["elements"] += n
            acc["p200_ss"] += p200_ss
            acc["pre_ss"] += pre_ss
            acc["event_ss"] += event_ss
            acc["dot"] += dot
    write_csv(output_dir / "parameter_diffs.csv", rows)
    groups = []
    for group, a in group_acc.items():
        groups.append({
            "group": group, "elements": int(a["elements"]),
            "delta_150_200_rms": math.sqrt(a["pre_ss"] / a["elements"]),
            "delta_200_250_rms": math.sqrt(a["event_ss"] / a["elements"]),
            "event_over_pre_delta_rms": math.sqrt(a["event_ss"] / max(a["pre_ss"], 1e-60)),
            "event_delta_over_param_rms": math.sqrt(a["event_ss"] / max(a["p200_ss"], 1e-60)),
            "pre_event_delta_cosine": a["dot"] / math.sqrt(max(a["pre_ss"] * a["event_ss"], 1e-60)),
        })
    write_csv(output_dir / "parameter_group_diffs.csv", groups)
    del states
    gc.collect()
    return {"tensor_count": len(rows), "common_keys": len(common), "steps": steps}


def fetch_wandb_history(run_path: str, output_dir: Path) -> dict:
    run = wandb.Api().run(run_path)
    keys = ["_step", "loss", "total_loss", "total_aux_loss", "grad_norm", "relative_grad_error",
            "fast_lr", "slow_lr", "loss_nan", "fast_grad_nan", "slow_grad_nan",
            "fast_update_nan", "slow_update_nan", "fast_param_nan", "slow_param_nan"]
    rows = list(run.scan_history(keys=keys, page_size=1000))
    df = pd.DataFrame(rows).sort_values("_step")
    df.to_csv(output_dir / "wandb_history.csv", index=False)
    window = df[(df._step >= 180) & (df._step <= 248)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(window._step, window.loss, label="loss")
    axes[0].plot(window._step, window.total_loss, label="mean LM loss", alpha=.8)
    axes[0].axvline(200, color="gray", ls="--"); axes[0].axvline(250, color="gray", ls="--")
    axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].semilogy(window._step, window.grad_norm, label="gradient norm")
    axes[1].plot(window._step, window.relative_grad_error, label="relative fast-gradient error")
    axes[1].set(xlabel="training step", ylabel="logged value"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(output_dir / "training_spike_timeline.png", dpi=180)
    plt.close(fig)
    finite = window[np.isfinite(window.grad_norm)]
    peak = finite.loc[finite.grad_norm.idxmax()]
    return {
        "run_path": run_path, "run_name": run.name, "state": run.state, "url": run.url,
        "history_rows": len(df), "peak_grad_step": int(peak._step), "peak_grad_norm": float(peak.grad_norm),
        "peak_loss": float(window.loss.max()), "peak_loss_step": int(window.loc[window.loss.idxmax(), "_step"]),
    }


class Collector:
    def __init__(self, model, step: int):
        self.model, self.step = model, step
        self.phase, self.episode = "", -1
        self.dynamic_rows, self.activation_rows, self.state_rows, self.top_rows = [], [], [], []
        self.layer_by_weight = {m.down_fast.weight.data_ptr(): i for i, m in enumerate(model.fast_modules())}
        self.original = forte._get_G
        self.handles = []
        self.activation_stage = ""
        self.activation_mask: torch.Tensor | None = None
        self.last_updates: dict[int, torch.Tensor] = {}

    def install(self):
        forte._get_G = self.get_g
        for layer, module in enumerate(self.model._causal_layers()):
            self.handles.append(module.register_forward_hook(self._activation_hook(layer)))

    def remove(self):
        forte._get_G = self.original
        for handle in self.handles:
            handle.remove()

    def _activation_hook(self, layer: int):
        def hook(_module, inputs, output):
            if not self.activation_stage or self.activation_mask is None or not torch.is_grad_enabled():
                return
            mask = self.activation_mask
            x, y = inputs[0].detach().float(), output.detach().float()
            if x.shape[0] == 2 * mask.shape[0]:
                mask = torch.repeat_interleave(mask, 2, dim=0)
            xv, yv = x[mask], y[mask]
            self.activation_rows.append({
                "step": self.step, "phase": self.phase, "stage": self.activation_stage,
                "episode": self.episode, "layer": layer, "tokens": int(mask.sum()),
                "input_rms": rms(xv), "output_rms": rms(yv),
                "residual_delta_rms": rms(yv - xv),
                "input_output_cosine": F.cosine_similarity(xv, yv, dim=-1).mean().item(),
            })
        return hook

    def set_activation(self, stage: str, mask: torch.Tensor):
        self.activation_stage, self.activation_mask = stage, mask.bool()

    def clear_activation(self):
        self.activation_stage, self.activation_mask = "", None

    def get_g(self, activations, output_grad, down_weight, valid_mask, *dynamic_args,
              eps: float, offset_alpha: float):
        raw_G, update = self.original(activations, output_grad, down_weight, valid_mask,
                                      *dynamic_args, eps=eps, offset_alpha=offset_alpha)
        layer = self.layer_by_weight[down_weight.data_ptr()]
        with torch.no_grad():
            offset, lr, offset_lr, gradient_logits, offset_logits = [x.detach().float() for x in dynamic_args]
            mask = valid_mask.bool()
            maskf = mask[..., None].float()
            count = maskf.sum(-2, keepdim=True).clamp_min(1)
            a = activations.detach().float() * maskf
            g = F.linear(output_grad.detach().float(), down_weight.detach().float().T) * maskf
            an = a * torch.rsqrt(a.square().sum(-2, keepdim=True) / count + eps**2)
            gn = g * torch.rsqrt(g.square().sum(-2, keepdim=True) / count + eps**2)
            gg, og = unit_softplus(gradient_logits), unit_softplus(offset_logits)
            base = -torch.einsum("blo,bli->boi", forte.gate_heads(gn, gg), an) * lr
            off_raw = -torch.einsum("blo,bli->boi", forte.gate_heads(offset, og), an) * offset_lr
            bn, on = norm_rows(base), norm_rows(off_raw)
            off = off_raw * offset_alpha * (bn / (on + eps))[:, None, None]
            total = base + off
            self.last_updates[layer] = total.detach()
            gs, os_ = masked_stats(gg, mask), masked_stats(og, mask)
            row = {
                "step": self.step, "phase": self.phase, "episode": self.episode, "layer": layer,
                "valid_tokens_mean": mask.sum(1).float().mean().item(),
                "activation_rms": rms(a[mask]), "output_grad_rms": rms(g[mask]), "offset_rms": rms(offset[mask]),
                "raw_G_norm": norm_rows(raw_G).mean().item(),
                "base_update_norm": bn.mean().item(), "offset_raw_norm": on.mean().item(),
                "offset_scaled_norm": norm_rows(off).mean().item(), "total_update_norm": norm_rows(total).mean().item(),
                "base_offset_cosine": cosine_rows(base, off).mean().item(),
                "total_over_base_norm": (norm_rows(total) / bn.clamp_min(1e-30)).mean().item(),
                "update_raw_G_cosine": cosine_rows(total, raw_G).mean().item(),
                "gradient_gate_lt_0.1": gg[mask].lt(.1).float().mean().item(),
                "gradient_gate_gt_2": gg[mask].gt(2).float().mean().item(),
                "offset_gate_lt_0.1": og[mask].lt(.1).float().mean().item(),
                "offset_gate_gt_2": og[mask].gt(2).float().mean().item(),
            }
            row.update({f"gradient_gate_{k}": v for k, v in gs.items()})
            row.update({f"offset_gate_{k}": v for k, v in os_.items()})
            self.dynamic_rows.append(row)
        return raw_G, update

    @torch.no_grad()
    def record_state(self):
        for layer, module in enumerate(self.model.fast_modules()):
            state, gb = module.state.float(), module.grad_buffer.float()
            update = self.last_updates.get(layer)
            self.state_rows.append({
                "step": self.step, "phase": self.phase, "episode": self.episode, "layer": layer,
                "state_norm": norm_rows(state).mean().item(), "state_rms": rms(state),
                "state_max_abs": state.abs().max().item(), "grad_buffer_norm": norm_rows(gb).mean().item(),
                "update_norm": norm_rows(update).mean().item() if update is not None else float("nan"),
                "state_update_cosine": cosine_rows(state, update).mean().item() if update is not None else float("nan"),
                "finite": bool(torch.isfinite(state).all() and torch.isfinite(gb).all()),
            })

    @torch.no_grad()
    def record_top(self, stage: str, tensor: torch.Tensor, mask: torch.Tensor):
        values = tensor.detach().float()[mask.bool()]
        self.top_rows.append({
            "step": self.step, "phase": self.phase, "stage": stage, "episode": self.episode,
            "tokens": int(mask.sum()), "rms": rms(values), "max_abs": values.abs().max().item(),
        })


def loss_and_grad(model, states, ids, assistant, valid, iterations: int, aux_weight: float):
    batch = states.shape[0]
    labels = ids[:, 1:]
    am, vm = assistant[:, 1:].float(), valid[:, 1:].float()
    nm = vm - am
    aw = am / am.sum(-1, keepdim=True).clamp_min(1) / batch
    nw = nm / nm.sum(-1, keepdim=True).clamp_min(1) / batch
    leaf = states.detach().reshape(-1, iterations, states.shape[-1]).requires_grad_(True)
    labels, aw, nw = (x.reshape(-1, iterations) for x in (labels, aw, nw))
    assistant_loss = states.new_zeros((), dtype=torch.float32)
    aux_loss = states.new_zeros((), dtype=torch.float32)
    for i in range(iterations):
        logits = model.lm_head(leaf[:, i]).float()
        raw = F.cross_entropy(logits, labels[:, i].contiguous(), reduction="none")
        al, nl = (raw * aw[:, i]).sum(), (raw * nw[:, i]).sum()
        (al + aux_weight * nl).backward()
        assistant_loss += al.detach(); aux_loss += nl.detach()
    return assistant_loss, aux_loss, leaf.grad.reshape_as(states).detach().to(states.dtype)


def first_pass(model, ids, assistant, valid, collector, iterations, aux_weight, final=False):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        collector.record_top("inferred_backbone", inferred, valid)
        embeddings = model.forward_embeddings(inferred, valid).detach()
        collector.record_top("lr_embeddings", embeddings, valid)
    collector.set_activation("train_first", valid)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_backbone(ids, mode=ForteMode.TRAIN_FIRST,
                                        embeddings=embeddings, embedding_mask=valid)
        states = model.forward_lm_states(hidden, mode=ForteMode.TRAIN_FIRST,
                                        logits_to_keep=slice(0, -1), embeddings=embeddings,
                                        embedding_mask=valid)
        collector.clear_activation()
        collector.record_top("lm_states", states, valid[:, :-1])
        al, nl, grad = loss_and_grad(model, states, ids, assistant, valid, iterations, aux_weight)
    if final:
        torch.autograd.backward(states, grad)
    else:
        torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(ForteMode.TRAIN_SECOND if final else ForteMode.TRAIN_FIRST)
    collector.record_state()
    return al.item(), nl.item()


def second_pass(model, ids, assistant, valid, collector, iterations, aux_weight):
    collector.set_activation("train_second", valid)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        double_ids = torch.repeat_interleave(ids, 2, dim=0)
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, valid)
        hidden = model.forward_backbone(double_ids, mode=ForteMode.TRAIN_SECOND,
                                        embeddings=embeddings, embedding_mask=valid)
        states = model.forward_lm_states(hidden, mode=ForteMode.TRAIN_SECOND,
                                        logits_to_keep=slice(0, -1), embeddings=embeddings,
                                        embedding_mask=valid)[::2]
        collector.clear_activation()
        al, nl, grad = loss_and_grad(model, states, ids, assistant, valid, iterations, aux_weight)
    torch.autograd.backward(states, grad)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(ForteMode.TRAIN_SECOND)
    collector.record_state()
    return al.item(), nl.item()


def gradient_rows(model, step: int) -> tuple[list[dict], dict]:
    rows, total_ss, total_n, missing = [], 0.0, 0, []
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(name)
            continue
        g = p.grad.detach().float()
        ss, n = g.square().sum().item(), g.numel()
        total_ss += ss; total_n += n
        rows.append({
            "step": step, "name": name, "family": tensor_family(name), "layer": tensor_layer(name),
            "elements": n, "grad_rms": math.sqrt(ss / n), "grad_norm": math.sqrt(ss),
            "grad_max_abs": g.abs().max().item(), "finite": bool(torch.isfinite(g).all()),
            "grad_param_cosine": (g * p.detach().float()).sum().item() /
                math.sqrt(max(ss * p.detach().float().square().sum().item(), 1e-60)),
        })
    return rows, {"global_grad_norm": math.sqrt(total_ss), "gradient_elements": total_n,
                  "missing_gradient_count": len(missing), "missing_gradients": missing}


def run_checkpoint(step: int, repo: str, batch_cpu: dict[str, torch.Tensor], output_dir: Path,
                   iterations: int, aux_weight: float) -> dict:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = load_checkpoint(repo, step, attention_kernel=None).cuda().eval()
    model.lm_head.to(torch.bfloat16)
    enable_gradient_checkpointing(model, True); model.train()
    batch = {k: v.cuda() for k, v in batch_cpu.items()}
    ids, assistant, valid = (batch[k] for k in ("input_ids", "assistant_mask", "attention_mask"))
    episodes, batch_size = ids.shape[1], ids.shape[0]
    model.init_state(batch_size, torch.device("cuda"))
    collector = Collector(model, step); collector.install()
    first_losses, second_losses = [], []
    try:
        for episode in range(episodes):
            collector.phase, collector.episode = "first", episode
            first_losses.append(first_pass(model, ids[:, episode], assistant[:, episode], valid[:, episode],
                                           collector, iterations, aux_weight))
            print(f"step {step} first {episode + 1}/{episodes}: {first_losses[-1][0]:.5f}", flush=True)
        model.finalize_state()
        final_buffer_norms = [m.final_grad_norm.mean().item() for m in model.fast_modules()]
        model.zero_grad(set_to_none=False)
        for episode in range(episodes - 1):
            collector.phase, collector.episode = "second", episode
            second_losses.append(second_pass(model, ids[:, episode], assistant[:, episode], valid[:, episode],
                                             collector, iterations, aux_weight))
            print(f"step {step} second {episode + 1}/{episodes}", flush=True)
        collector.phase, collector.episode = "second_final", episodes - 1
        second_losses.append(first_pass(model, ids[:, -1], assistant[:, -1], valid[:, -1],
                                        collector, iterations, aux_weight, final=True))
        relative_error = model.relative_grad_error().mean().item()
        grads, grad_summary = gradient_rows(model, step)
    finally:
        collector.remove()
    write_csv(output_dir / f"dynamic_metrics_step{step}.csv", collector.dynamic_rows)
    write_csv(output_dir / f"activation_metrics_step{step}.csv", collector.activation_rows)
    write_csv(output_dir / f"state_metrics_step{step}.csv", collector.state_rows)
    write_csv(output_dir / f"top_activation_metrics_step{step}.csv", collector.top_rows)
    write_csv(output_dir / f"outer_gradients_step{step}.csv", grads)
    result = {
        "step": step, "first_losses": first_losses, "second_losses": second_losses,
        "first_mean_lm_loss": float(np.mean([x[0] for x in first_losses])),
        "first_mean_aux_loss": float(np.mean([x[1] for x in first_losses])),
        "final_grad_buffer_norms": final_buffer_norms, "relative_grad_error": relative_error,
        **grad_summary, "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    del model, batch, ids, assistant, valid
    gc.collect(); torch.cuda.empty_cache()
    return result


def make_comparison_plots(output_dir: Path, steps: list[int]) -> None:
    dynamic = pd.concat([pd.read_csv(output_dir / f"dynamic_metrics_step{s}.csv") for s in steps])
    first = dynamic[dynamic.phase == "first"]
    metrics = ["gradient_gate_mean", "offset_gate_mean", "total_update_norm", "base_offset_cosine"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        for step in steps:
            grid = first[first.step == step].groupby("episode")[metric].mean()
            ax.plot(grid.index, grid.values, label=f"step {step}")
        ax.set(title=metric.replace("_", " "), xlabel="episode"); ax.grid(alpha=.2); ax.legend()
    fig.savefig(output_dir / "dynamic_trajectory_comparison.png", dpi=180); plt.close(fig)

    states = pd.concat([pd.read_csv(output_dir / f"state_metrics_step{s}.csv") for s in steps])
    states = states[states.phase == "first"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric in zip(axes, ("state_norm", "grad_buffer_norm")):
        for step in steps:
            grid = states[states.step == step].groupby("episode")[metric].mean()
            ax.plot(grid.index, grid.values, label=f"step {step}")
        ax.set(title=metric.replace("_", " "), xlabel="episode"); ax.grid(alpha=.2); ax.legend()
    fig.savefig(output_dir / "state_trajectory_comparison.png", dpi=180); plt.close(fig)

    grads = pd.concat([pd.read_csv(output_dir / f"outer_gradients_step{s}.csv") for s in steps])
    family = grads.groupby(["step", "family"]).apply(
        lambda x: math.sqrt(float((x.grad_norm ** 2).sum())), include_groups=False
    ).rename("norm").reset_index()
    pivot = family.pivot(index="family", columns="step", values="norm").fillna(0)
    pivot["max"] = pivot.max(axis=1); pivot = pivot.sort_values("max", ascending=False).head(16).drop(columns="max")
    ax = pivot.plot.bar(figsize=(13, 6), logy=True)
    ax.set(ylabel="outer gradient norm", title="Same-data outer gradients by parameter family")
    ax.grid(axis="y", alpha=.2); ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "outer_gradient_family_comparison.png", dpi=180); plt.close(ax.figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="aklein4/horizon-v2_forte-offset")
    ap.add_argument("--steps", nargs=2, type=int, default=[200, 250])
    ap.add_argument("--baseline-step", type=int, default=150)
    ap.add_argument("--trajectories", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--data-config", type=Path, default=SRC / "configs/data/300k-horizons-llama3.yaml")
    ap.add_argument("--output-dir", type=Path, default=SRC / "local_data/horizon_v2_forte_offset_spike_analysis")
    ap.add_argument("--wandb-run", default="aklein4/horizon-v2/fyqw3ckb")
    ap.add_argument("--aux-loss-weight", type=float, default=.1)
    ap.add_argument("--num-logit-iterations", type=int, default=4)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(42); np.random.seed(42); torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = SRC / "local_data/checkpoints" / args.repo.replace("/", "--")

    wandb_summary = fetch_wandb_history(args.wandb_run, args.output_dir)
    param_summary = analyze_parameter_diffs(checkpoint_root, [args.baseline_step, *args.steps], args.output_dir)

    cfg = OmegaConf.load(args.data_config)
    stream = datasets.load_dataset(cfg.dataset.url, **cfg.dataset.kwargs)
    iterator = iter(stream)
    raw = [next(iterator) for _ in range(args.trajectories)]
    kwargs = OmegaConf.to_container(cfg.collator.kwargs, resolve=True)
    kwargs["cluster_length"] = args.episodes
    batch = HorizonCollator(**kwargs)(raw)
    batch_cpu = {k: batch[k].cpu() for k in ("input_ids", "assistant_mask", "attention_mask")}
    data_summary = {
        "dataset": cfg.dataset.url, "trajectories": args.trajectories, "episodes": args.episodes,
        "max_length": kwargs["max_length"], "sources": [r.get("source") for r in raw],
        "cluster_num_tokens": [r.get("num_tokens") for r in raw],
        "valid_tokens": int(batch_cpu["attention_mask"].sum()),
        "assistant_tokens": int(batch_cpu["assistant_mask"].sum()),
    }
    checkpoint_results = [
        run_checkpoint(step, args.repo, batch_cpu, args.output_dir,
                       args.num_logit_iterations, args.aux_loss_weight)
        for step in args.steps
    ]
    make_comparison_plots(args.output_dir, args.steps)
    metadata = {
        "repo": args.repo, "steps": args.steps, "baseline_step_for_parameter_rate": args.baseline_step,
        "wandb": wandb_summary, "parameters": param_summary, "data": data_summary,
        "checkpoint_results": checkpoint_results, "aux_loss_weight": args.aux_loss_weight,
        "num_logit_iterations": args.num_logit_iterations, "gradient_checkpointing": True,
        "torch_xla_used": False, "device": torch.cuda.get_device_name(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
