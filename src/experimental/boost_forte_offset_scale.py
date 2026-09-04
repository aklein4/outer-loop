"""Measure baseline offset/base RMS and test a 5x realized offset at every update.

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

SCENARIOS = (("baseline", 1.0), ("offset_rms_x5", 5.0))
OBJECTIVES = ("assistant", "nonassistant", "weighted_total")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class OffsetScaleRule:
    def __init__(self, model):
        self.original = forte._get_G
        self.scale = 1.0

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
        if self.scale == 1:
            return raw_g, total
        _, _, lr, _, gradient_logits, _ = [value.float() for value in dynamic_args]
        mask = valid_mask.bool()[..., None].float()
        count = mask.sum(-2, keepdim=True).clamp_min(1)
        a = activations.float() * mask
        g = F.linear(output_grad.float() * mask, down_weight.float().T)
        a_norm = a * torch.rsqrt(a.square().sum(-2, keepdim=True) / count + eps**2)
        g_norm = g * torch.rsqrt(g.square().sum(-2, keepdim=True) / count + eps**2)
        base = -torch.einsum(
            "blo,bli->boi",
            forte.gate_heads(g_norm, unit_softplus(gradient_logits)),
            a_norm,
        ) * lr
        realized_offset = total.float() - base
        return raw_g, base + self.scale * realized_offset


def loss_and_grad(model, states, ids, assistant, valid, logical_batch, iterations, aux_weight):
    labels = ids[:, 1:]
    assistant, valid = assistant[:, 1:].float(), valid[:, 1:].float()
    nonassistant = valid - assistant
    aw = assistant / assistant.sum(-1, keepdim=True).clamp_min(1)
    nw = nonassistant / nonassistant.sum(-1, keepdim=True).clamp_min(1)
    weights = aw + aux_weight * nw
    leaf = states.detach().requires_grad_(True)
    raw_full = torch.empty_like(labels, dtype=torch.float32)
    for iteration in range(iterations):
        logits = model.lm_head(leaf[:, iteration::iterations]).float()
        raw = F.cross_entropy(
            logits.flatten(0, 1), labels[:, iteration::iterations].flatten(),
            reduction="none",
        ).reshape_as(labels[:, iteration::iterations])
        (raw * weights[:, iteration::iterations]).sum().div(logical_batch).backward()
        raw_full[:, iteration::iterations] = raw.detach()
    return (
        (raw_full * aw).sum(-1),
        (raw_full * nw).sum(-1),
        leaf.grad.detach().to(states.dtype),
    )


def recurrent_step(model, ids, assistant, valid, batch_size, iterations, aux_weight):
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
        assistant_loss, nonassistant_loss, grad = loss_and_grad(
            model, states, ids, assistant, valid, batch_size, iterations, aux_weight,
        )
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(ForteMode.TRAIN_FIRST)
    return assistant_loss.cpu().numpy(), nonassistant_loss.cpu().numpy()


def summarize_overall(rows: list[dict]) -> list[dict]:
    output = []
    for objective in OBJECTIVES:
        trajectory_means = {}
        for trajectory in sorted({row["trajectory"] for row in rows}):
            for scenario, _ in SCENARIOS:
                values = [row[f"{objective}_loss"] for row in rows
                          if row["trajectory"] == trajectory and row["scenario"] == scenario
                          and row["t"] > 0]
                trajectory_means[trajectory, scenario] = np.mean(values)
        baseline = np.asarray([trajectory_means[t, "baseline"] for t in range(16)])
        boosted = np.asarray([trajectory_means[t, "offset_rms_x5"] for t in range(16)])
        delta = boosted - baseline
        se = delta.std(ddof=1) / math.sqrt(len(delta))
        output.append({
            "objective": objective, "trajectories": len(delta),
            "baseline_loss": baseline.mean(), "boosted_loss": boosted.mean(),
            "delta_loss": delta.mean(), "delta_percent": 100 * delta.mean() / baseline.mean(),
            "ci95_low": delta.mean() - 1.96 * se,
            "ci95_high": delta.mean() + 1.96 * se,
            "trajectories_worse": int((delta > 0).sum()),
        })
    return output


def summarize_by_t(rows: list[dict]) -> list[dict]:
    output = []
    for t in sorted({row["t"] for row in rows}):
        for objective in OBJECTIVES:
            baseline = np.asarray([row[f"{objective}_loss"] for row in rows
                                   if row["t"] == t and row["scenario"] == "baseline"])
            boosted = np.asarray([row[f"{objective}_loss"] for row in rows
                                  if row["t"] == t and row["scenario"] == "offset_rms_x5"])
            delta = boosted - baseline
            se = delta.std(ddof=1) / math.sqrt(len(delta))
            output.append({
                "t": t, "objective": objective,
                "baseline_mean": baseline.mean(), "boosted_mean": boosted.mean(),
                "delta_loss": delta.mean(),
                "delta_percent": 100 * delta.mean() / baseline.mean(),
                "ci95_low": delta.mean() - 1.96 * se,
                "ci95_high": delta.mean() + 1.96 * se,
            })
    return output


def summarize_ratio(diagnostics: Path) -> list[dict]:
    with diagnostics.open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        # Frobenius norm and RMS ratios are identical because both matrices
        # have the same number of elements.
        grouped[int(row["chunk"])].append(float(row["offset_over_base"]))
    output = []
    for t, values in sorted(grouped.items()):
        x = np.asarray(values)
        output.append({
            "t": t, "observations": len(x), "mean_ratio": x.mean(),
            "median_ratio": np.median(x), "p25": np.quantile(x, .25),
            "p75": np.quantile(x, .75), "fraction_at_cap": np.mean(x > .249),
        })
    return output


def plots(ratio: list[dict], by_t: list[dict], output_dir: Path) -> None:
    t = np.asarray([row["t"] for row in ratio])
    mean = np.asarray([row["mean_ratio"] for row in ratio])
    median = np.asarray([row["median_ratio"] for row in ratio])
    p25 = np.asarray([row["p25"] for row in ratio])
    p75 = np.asarray([row["p75"] for row in ratio])
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.fill_between(t, p25, p75, alpha=.2, color="#4472c4", label="layer×trajectory IQR")
    ax.plot(t, mean, color="#4472c4", label="mean")
    ax.plot(t, median, color="#ed7d31", label="median")
    ax.axhline(.25, color="black", ls="--", lw=.8, label="nominal cap")
    ax.set(xlabel="t", ylabel="RMS(realized offset) / RMS(no-offset update)",
           title="Baseline realized offset ratio through recurrence")
    ax.legend()
    fig.savefig(output_dir / "baseline_offset_rms_ratio_through_t.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    colors = {"baseline": "#4472c4", "offset_rms_x5": "#c00000"}
    for objective, ls in (("assistant", "-"), ("nonassistant", "--")):
        selected = [row for row in by_t if row["objective"] == objective]
        tx = np.asarray([row["t"] for row in selected])
        for scenario, field in (("baseline", "baseline_mean"), ("offset_rms_x5", "boosted_mean")):
            axes[0].plot(tx, [row[field] for row in selected], color=colors[scenario], ls=ls,
                         label=f"{scenario}, {objective}")
    assistant = [row for row in by_t if row["objective"] == "assistant"]
    tx = np.asarray([row["t"] for row in assistant])
    delta = np.asarray([row["delta_percent"] for row in assistant])
    low = np.asarray([100 * row["ci95_low"] / row["baseline_mean"] for row in assistant])
    high = np.asarray([100 * row["ci95_high"] / row["baseline_mean"] for row in assistant])
    axes[1].fill_between(tx, low, high, color="#c00000", alpha=.2, label="paired 95% CI")
    axes[1].plot(tx, delta, color="#c00000", label="assistant Δloss")
    axes[1].axhline(0, color="black", lw=.8)
    axes[0].set(ylabel="mean loss", title="Effect of 5× realized offset at every update")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set(xlabel="t", ylabel="assistant Δloss (%)")
    axes[1].legend()
    fig.savefig(output_dir / "offset_rms_x5_losses_through_t.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="aklein4/horizon-v2_forte-delta")
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--data-config", type=Path,
                        default=SRC / "configs/data/300k-horizons-llama3.yaml")
    parser.add_argument("--diagnostics", type=Path, default=SRC / "local_data/"
                        "horizon_v2_forte_delta_step1000_ablation/"
                        "baseline_component_diagnostics.csv")
    parser.add_argument("--output-dir", type=Path, default=SRC / "local_data/"
                        "horizon_v2_forte_delta_step1000_offset_rms_x5")
    parser.add_argument("--aux-loss-weight", type=float, default=.1)
    parser.add_argument("--num-logit-iterations", type=int, default=4)
    args = parser.parse_args()
    assert torch.cuda.is_available() and args.trajectories % args.batch_size == 0
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
    collator = HorizonCollator(**kwargs)
    model = load_checkpoint(args.checkpoint, args.step, attention_kernel=None).cuda().eval()
    model.lm_head.to(torch.bfloat16)
    enable_gradient_checkpointing(model, True)
    model.train()
    rule = OffsetScaleRule(model)
    rule.install()
    loss_rows = []
    try:
        for start in range(0, args.trajectories, args.batch_size):
            batch = collator(raw[start:start + args.batch_size])
            ids, assistant, valid = (
                batch[key].cuda() for key in ("input_ids", "assistant_mask", "attention_mask")
            )
            for scenario, scale in SCENARIOS:
                rule.scale = scale
                model.init_state(args.batch_size, torch.device("cuda"))
                for t in range(args.chunks):
                    al, nl = recurrent_step(
                        model, ids[:, t], assistant[:, t], valid[:, t], args.batch_size,
                        args.num_logit_iterations, args.aux_loss_weight,
                    )
                    for local in range(args.batch_size):
                        loss_rows.append({
                            "trajectory": start + local, "batch": start // args.batch_size,
                            "t": t, "scenario": scenario,
                            "assistant_loss": float(al[local]),
                            "nonassistant_loss": float(nl[local]),
                            "weighted_total_loss": float(al[local] + args.aux_loss_weight * nl[local]),
                        })
                    print(f"batch {start // args.batch_size + 1}/"
                          f"{args.trajectories // args.batch_size} {scenario} "
                          f"t {t + 1}/{args.chunks}", flush=True)
                torch.cuda.empty_cache()
    finally:
        rule.remove()

    overall = summarize_overall(loss_rows)
    by_t = summarize_by_t(loss_rows)
    ratio = summarize_ratio(args.diagnostics)
    write_csv(args.output_dir / "losses.csv", loss_rows)
    write_csv(args.output_dir / "overall_summary.csv", overall)
    write_csv(args.output_dir / "losses_by_t.csv", by_t)
    write_csv(args.output_dir / "baseline_offset_rms_ratio_by_t.csv", ratio)
    plots(ratio, by_t, args.output_dir)
    metadata = {
        "checkpoint": args.checkpoint, "step": args.step,
        "dataset": config.dataset.url, "trajectories": args.trajectories,
        "batch_size": args.batch_size, "chunks": args.chunks,
        "boost_definition": "multiply the realized norm-capped offset contribution by 5 immediately before adding it to the unchanged no-offset update, at every recurrent update",
        "ratio_definition": "RMS(realized norm-capped offset contribution) / RMS(no-offset preconditioned, gradient-gated, learned-LR update); equal to their Frobenius norm ratio",
        "ratio_source": str(args.diagnostics),
        "aux_loss_weight": args.aux_loss_weight,
        "num_logit_iterations": args.num_logit_iterations,
        "torch_xla_used": False, "cuda_device": torch.cuda.get_device_name(),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"overall": overall}, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
