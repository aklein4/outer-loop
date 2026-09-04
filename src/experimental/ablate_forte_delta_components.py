"""Matched recurrent component ablations for the Forte-delta architecture.

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


SCENARIOS = (
    "baseline",
    "no_offset_update",
    "only_offset_update",
    "offset_without_Mx",
    "offset_without_target",
    "offset_gate_one",
    "offset_gate_token_mean",
    "offset_gate_batch_mean",
    "uniform_offset_lr",
    "gradient_gate_one",
    "uniform_base_lr",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def row_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(1).norm(dim=1)


def row_cos(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xf, yf = x.float().flatten(1), y.float().flatten(1)
    return (xf*yf).sum(1) / (xf.norm(dim=1)*yf.norm(dim=1)).clamp_min(1e-30)


class ComponentRule:
    def __init__(self, model, logical_batch: int, scenarios=SCENARIOS):
        self.model, self.logical_batch = model, logical_batch
        self.scenarios = tuple(scenarios)
        self.original = forte._get_G
        self.layer_by_weight = {m.down_fast.weight.data_ptr(): (i, m)
                                for i, m in enumerate(model.fast_modules())}
        self.chunk = -1
        self.rows: list[dict] = []

    def install(self):
        forte._get_G = self.get_g

    def remove(self):
        forte._get_G = self.original

    @staticmethod
    def batch_mean(gate: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        w = valid.bool()[..., None].to(gate.dtype)
        mean = (gate*w).sum((0, 1), keepdim=True) / w.sum((0, 1), keepdim=True).clamp_min(1)
        return mean.expand_as(gate)

    def get_g(self, activations, output_grad, down_weight, valid_mask, *dynamic_args,
              eps: float, offset_alpha: float):
        layer, module = self.layer_by_weight[down_weight.data_ptr()]
        valid_mask = valid_mask.bool()
        Mx, target, lr, offset_lr, gradient_logits, offset_logits = [x.float() for x in dynamic_args]
        mask = valid_mask[..., None].float()
        count = mask.sum(-2, keepdim=True).clamp_min(1)
        a = activations.float()*mask
        g = F.linear(output_grad.float()*mask, down_weight.float().T)
        raw_G = torch.einsum("blo,bli->boi", g, a)
        a_norm = a*torch.rsqrt(a.square().sum(-2, keepdim=True)/count + eps**2)
        g_norm = g*torch.rsqrt(g.square().sum(-2, keepdim=True)/count + eps**2)
        gradient_gate = unit_softplus(gradient_logits)
        offset_gate = unit_softplus(offset_logits)
        nominal_lr = module.fast_dynamic_lr.base_lr/module.fast_weight_size
        b = self.logical_batch
        updates = []
        for scenario_i, scenario in enumerate(self.scenarios):
            sl = slice(scenario_i*b, (scenario_i+1)*b)
            gg = gradient_gate[sl]
            if scenario == "gradient_gate_one":
                gg = torch.ones_like(gg)
            base_lr = torch.full_like(lr, nominal_lr) if scenario == "uniform_base_lr" else lr
            base = -torch.einsum("blo,bli->boi", forte.gate_heads(g_norm[sl], gg), a_norm[sl])*base_lr

            delta = target[sl]-Mx[sl]
            if scenario == "offset_without_Mx":
                delta = target[sl]
            elif scenario == "offset_without_target":
                delta = -Mx[sl]
            og = offset_gate[sl]
            if scenario == "offset_gate_one":
                og = torch.ones_like(og)
            elif scenario == "offset_gate_token_mean":
                og = og.mean(-1, keepdim=True).expand_as(og)
            elif scenario == "offset_gate_batch_mean":
                og = self.batch_mean(og, valid_mask[sl])
            off_lr = torch.full_like(offset_lr, nominal_lr) if scenario == "uniform_offset_lr" else offset_lr
            off_raw = torch.einsum("blo,bli->boi", delta*og, a_norm[sl])*off_lr
            base_norm, off_norm = row_norm(base)[:, None, None], row_norm(off_raw)[:, None, None]
            cap = offset_alpha*torch.tanh(off_norm/(base_norm+eps))*base_norm/(off_norm+eps)
            off_realized = off_raw*cap
            if scenario == "no_offset_update":
                total = base
            elif scenario == "only_offset_update":
                total = off_realized
            else:
                total = base+off_realized
            updates.append(total)

            if scenario == "baseline":
                with torch.no_grad():
                    target_norm = target[sl].float().square().sum(-1).sqrt()
                    mx_norm = Mx[sl].float().square().sum(-1).sqrt()
                    delta_norm = delta.float().square().sum(-1).sqrt()
                    target_mx_cos = F.cosine_similarity(target[sl].float(), Mx[sl].float(), dim=-1)
                    for j in range(b):
                        vm = valid_mask[sl][j]
                        self.rows.append({
                            "trajectory_in_batch": j, "chunk": self.chunk, "layer": layer,
                            "valid_tokens": int(vm.sum()),
                            "base_update_norm": row_norm(base)[j].item(),
                            "offset_raw_norm": row_norm(off_raw)[j].item(),
                            "offset_realized_norm": row_norm(off_realized)[j].item(),
                            "total_update_norm": row_norm(total)[j].item(),
                            "offset_over_base": (row_norm(off_realized)[j]/row_norm(base)[j].clamp_min(1e-30)).item(),
                            "base_offset_cosine": row_cos(base, off_realized)[j].item(),
                            "cap_fraction_of_base": (offset_alpha*torch.tanh(off_norm[j]/(base_norm[j]+eps))).item(),
                            "target_token_norm": target_norm[j][vm].mean().item(),
                            "Mx_token_norm": mx_norm[j][vm].mean().item(),
                            "delta_token_norm": delta_norm[j][vm].mean().item(),
                            "target_Mx_cosine": target_mx_cos[j][vm].mean().item(),
                            "gradient_gate_mean": gradient_gate[sl][j][vm].mean().item(),
                            "gradient_gate_std": gradient_gate[sl][j][vm].std(unbiased=False).item(),
                            "offset_gate_mean": offset_gate[sl][j][vm].mean().item(),
                            "offset_gate_std": offset_gate[sl][j][vm].std(unbiased=False).item(),
                        })
        return raw_G, torch.cat(updates, 0)


def loss_and_grad(model, states, ids, assistant, valid, logical_batch, iterations, aux_weight):
    labels = ids[:, 1:]
    am, vm = assistant[:, 1:].float(), valid[:, 1:].float()
    nm = vm-am
    aw = am/am.sum(-1, keepdim=True).clamp_min(1)
    nw = nm/nm.sum(-1, keepdim=True).clamp_min(1)
    combined = aw+aux_weight*nw
    leaf = states.detach().requires_grad_(True)
    raw_full = torch.empty_like(labels, dtype=torch.float32)
    for i in range(iterations):
        logits = model.lm_head(leaf[:, i::iterations]).float()
        raw = F.cross_entropy(logits.flatten(0, 1), labels[:, i::iterations].flatten(),
                              reduction="none").reshape_as(labels[:, i::iterations])
        (raw*combined[:, i::iterations]).sum().div(logical_batch).backward()
        raw_full[:, i::iterations] = raw.detach()
    raw = raw_full
    return ((raw*aw).sum(-1), (raw*nw).sum(-1),
            leaf.grad.detach().to(states.dtype))


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


def paired_summary(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        if r["chunk"] == 0:
            continue
        for objective in ("assistant", "nonassistant", "weighted_total"):
            by[r["trajectory"], r["scenario"], objective].append(r[f"{objective}_loss"])
    means = {k: np.mean(v) for k, v in by.items()}
    trajectories = sorted({r["trajectory"] for r in rows})
    out = []
    for scenario in SCENARIOS[1:]:
        for objective in ("assistant", "nonassistant", "weighted_total"):
            base = np.array([means[t, "baseline", objective] for t in trajectories])
            abl = np.array([means[t, scenario, objective] for t in trajectories])
            delta = abl-base
            se = delta.std(ddof=1)/math.sqrt(len(delta))
            out.append({
                "scenario": scenario, "objective": objective, "trajectories": len(delta),
                "baseline_loss": base.mean(), "ablated_loss": abl.mean(),
                "delta_loss": delta.mean(), "delta_percent": 100*delta.mean()/base.mean(),
                "ci95_low": delta.mean()-1.96*se, "ci95_high": delta.mean()+1.96*se,
                "positive_trajectories": int((delta > 0).sum()),
            })
    return out


def chunk_summary(rows: list[dict]) -> list[dict]:
    out = []
    for lo, hi, label in ((1, 8, "early_1_8"), (9, 31, "middle_9_31"), (32, 63, "late_32_63")):
        selected = [r for r in rows if lo <= r["chunk"] <= hi]
        by = defaultdict(list)
        for r in selected:
            for objective in ("assistant", "nonassistant", "weighted_total"):
                by[r["trajectory"], r["scenario"], objective].append(r[f"{objective}_loss"])
        trajectories = sorted({r["trajectory"] for r in selected})
        for scenario in SCENARIOS[1:]:
            for objective in ("assistant", "nonassistant", "weighted_total"):
                b = np.array([np.mean(by[t, "baseline", objective]) for t in trajectories])
                a = np.array([np.mean(by[t, scenario, objective]) for t in trajectories])
                d = a-b
                out.append({"range": label, "scenario": scenario, "objective": objective,
                            "baseline_loss": b.mean(), "delta_loss": d.mean(),
                            "delta_percent": 100*d.mean()/b.mean()})
    return out


def plot_summary(summary: list[dict], output: Path) -> None:
    scenarios = list(SCENARIOS[1:])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    for ax, objective in zip(axes, ("assistant", "nonassistant")):
        rs = {r["scenario"]: r for r in summary if r["objective"] == objective}
        y = np.array([rs[s]["delta_percent"] for s in scenarios])
        denom = np.array([rs[s]["baseline_loss"] for s in scenarios])
        lo = np.array([100*rs[s]["ci95_low"]/denom[i] for i, s in enumerate(scenarios)])
        hi = np.array([100*rs[s]["ci95_high"]/denom[i] for i, s in enumerate(scenarios)])
        ax.bar(range(len(scenarios)), y, color="#4472c4")
        ax.errorbar(range(len(scenarios)), y, yerr=[y-lo, hi-y], fmt="none", color="black", capsize=3)
        ax.axhline(0, color="black", lw=.8); ax.set_title(objective.title()+" loss")
        ax.set_ylabel("paired Δloss (%)"); ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], rotation=30, ha="right", fontsize=8)
    fig.savefig(output, dpi=180); plt.close(fig)

    small = [s for s in scenarios if s not in ("only_offset_update", "gradient_gate_one")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, objective in zip(axes, ("assistant", "nonassistant")):
        rs = {r["scenario"]: r for r in summary if r["objective"] == objective}
        y = np.array([rs[s]["delta_percent"] for s in small])
        denom = np.array([rs[s]["baseline_loss"] for s in small])
        lo = np.array([100*rs[s]["ci95_low"]/denom[i] for i, s in enumerate(small)])
        hi = np.array([100*rs[s]["ci95_high"]/denom[i] for i, s in enumerate(small)])
        ax.bar(range(len(small)), y, color="#4472c4")
        ax.errorbar(range(len(small)), y, yerr=[y-lo, hi-y], fmt="none", color="black", capsize=3)
        ax.axhline(0, color="black", lw=.8); ax.set_title(objective.title()+" loss: small effects")
        ax.set_ylabel("paired Δloss (%)"); ax.set_xticks(range(len(small)))
        ax.set_xticklabels([s.replace("_", "\n") for s in small], rotation=30, ha="right", fontsize=8)
    fig.savefig(output.with_name("ablation_summary_small_effects.png"), dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="aklein4/horizon-v2_forte-delta")
    ap.add_argument("--step", type=int, default=200)
    ap.add_argument("--trajectories", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=64)
    ap.add_argument("--data-config", type=Path, default=SRC/"configs/data/300k-horizons-llama3.yaml")
    ap.add_argument("--output-dir", type=Path, default=SRC/"local_data/horizon_v2_forte_delta_step200_ablation")
    ap.add_argument("--aux-loss-weight", type=float, default=.1)
    ap.add_argument("--num-logit-iterations", type=int, default=4)
    ap.add_argument(
        "--scenario-group-size", type=int, default=len(SCENARIOS),
        help="Run this many scenarios concurrently; smaller groups reduce peak CUDA memory.",
    )
    args = ap.parse_args()
    if args.trajectories % args.batch_size:
        raise ValueError("trajectories must be divisible by batch size")
    if not 1 <= args.scenario_group_size <= len(SCENARIOS):
        raise ValueError(f"scenario-group-size must be in [1, {len(SCENARIOS)}]")
    assert torch.cuda.is_available()
    torch.manual_seed(42); np.random.seed(42); torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(args.checkpoint, args.step, attention_kernel=None).cuda().eval()
    model.lm_head.to(torch.bfloat16); enable_gradient_checkpointing(model, True); model.train()
    cfg = OmegaConf.load(args.data_config)
    stream = datasets.load_dataset(cfg.dataset.url, **cfg.dataset.kwargs)
    iterator = iter(stream); raw = [next(iterator) for _ in range(args.trajectories)]
    kwargs = OmegaConf.to_container(cfg.collator.kwargs, resolve=True); kwargs["cluster_length"] = args.chunks
    collator = HorizonCollator(**kwargs)
    loss_rows, diagnostic_rows = [], []
    for start in range(0, args.trajectories, args.batch_size):
        batch = collator(raw[start:start+args.batch_size])
        base_ids, base_assistant, base_valid = (batch[k].cuda() for k in ("input_ids", "assistant_mask", "attention_mask"))
        scenario_groups = [SCENARIOS[i:i+args.scenario_group_size]
                           for i in range(0, len(SCENARIOS), args.scenario_group_size)]
        for group_i, scenarios in enumerate(scenario_groups):
            ids = torch.cat([base_ids]*len(scenarios), 0)
            assistant = torch.cat([base_assistant]*len(scenarios), 0)
            valid = torch.cat([base_valid]*len(scenarios), 0)
            model.init_state(args.batch_size*len(scenarios), torch.device("cuda"))
            rule = ComponentRule(model, args.batch_size, scenarios); rule.install()
            try:
                for chunk in range(args.chunks):
                    rule.chunk = chunk
                    al, nl = recurrent_step(model, ids[:, chunk], assistant[:, chunk], valid[:, chunk],
                                            args.batch_size, args.num_logit_iterations, args.aux_loss_weight)
                    for si, scenario in enumerate(scenarios):
                        for t in range(args.batch_size):
                            idx = si*args.batch_size+t
                            loss_rows.append({
                                "trajectory": start+t, "batch": start//args.batch_size, "chunk": chunk,
                                "scenario": scenario, "assistant_loss": float(al[idx]),
                                "nonassistant_loss": float(nl[idx]),
                                "weighted_total_loss": float(al[idx]+args.aux_loss_weight*nl[idx]),
                                "valid_tokens": int(base_valid[t, chunk].sum()),
                                "assistant_tokens": int(base_assistant[t, chunk].sum()),
                            })
                    print(f"batch {start//args.batch_size+1}/{args.trajectories//args.batch_size} "
                          f"group {group_i+1}/{len(scenario_groups)} chunk {chunk+1}/{args.chunks} "
                          f"{scenarios[0]} assistant={al[:args.batch_size].mean():.5f}", flush=True)
            finally:
                rule.remove()
            for r in rule.rows:
                r["trajectory"] = start+r.pop("trajectory_in_batch")
            diagnostic_rows.extend(rule.rows)
            del ids, assistant, valid
            torch.cuda.empty_cache()
        del base_ids, base_assistant, base_valid, batch
    write_csv(args.output_dir/"ablation_losses.csv", loss_rows)
    summary = paired_summary(loss_rows); write_csv(args.output_dir/"ablation_summary.csv", summary)
    write_csv(args.output_dir/"ablation_by_chunk_range.csv", chunk_summary(loss_rows))
    write_csv(args.output_dir/"baseline_component_diagnostics.csv", diagnostic_rows)
    plot_summary(summary, args.output_dir/"ablation_summary.png")
    metadata = {
        "checkpoint": args.checkpoint, "step": args.step, "dataset": cfg.dataset.url,
        "data_config": str(args.data_config), "trajectories": args.trajectories,
        "batch_size": args.batch_size, "chunks": args.chunks, "max_length": cfg.collator.kwargs.max_length,
        "scenarios": list(SCENARIOS), "aux_loss_weight": args.aux_loss_weight,
        "num_logit_iterations": args.num_logit_iterations, "gradient_checkpointing": True,
        "scenario_group_size": args.scenario_group_size,
        "torch_xla_used": False, "cuda_device": torch.cuda.get_device_name(),
        "peak_cuda_gb": torch.cuda.max_memory_allocated()/2**30,
        "only_offset_definition": "retain realized offset contribution with cap computed against the original base branch, but omit base contribution",
        "uniform_lr_definition": "replace selected learned matrix LR by base_lr/fast_weight_size at every coordinate",
    }
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()
