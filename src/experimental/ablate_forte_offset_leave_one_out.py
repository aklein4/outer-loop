"""Replace each realized Forte offset update with its leave-one-out batch mean.

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


SCENARIOS = ("baseline", "offset_leave_one_out_batch_mean")
OBJECTIVES = ("assistant", "nonassistant", "weighted_total")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class LeaveOneOutOffsetRule:
    def __init__(self, model, batch_size: int):
        if batch_size < 2:
            raise ValueError("leave-one-out replacement requires batch_size >= 2")
        self.original = forte._get_G
        self.batch_size = batch_size
        self.enabled = False
        self.t = -1
        self.batch = -1
        self.module_by_weight = {
            module.down_fast.weight.data_ptr(): layer
            for layer, module in enumerate(model.fast_modules())
        }
        self.geometry_rows: list[dict] = []

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
        if not self.enabled:
            return raw_g, total

        if total.shape[0] != self.batch_size:
            raise RuntimeError(
                f"expected physical batch {self.batch_size}, got {total.shape[0]}"
            )

        _, _, lr, _, gradient_logits, _ = [value.float() for value in dynamic_args]
        mask = valid_mask.bool()[..., None].float()
        count = mask.sum(-2, keepdim=True).clamp_min(1)
        activations_f = activations.float() * mask
        output_g = F.linear(output_grad.float() * mask, down_weight.float().T)
        a_norm = activations_f * torch.rsqrt(
            activations_f.square().sum(-2, keepdim=True) / count + eps**2
        )
        g_norm = output_g * torch.rsqrt(
            output_g.square().sum(-2, keepdim=True) / count + eps**2
        )
        base = -torch.einsum(
            "blo,bli->boi",
            forte.gate_heads(g_norm, unit_softplus(gradient_logits)),
            a_norm,
        ) * lr
        realized_offset = total.float() - base
        loo_offset = (
            realized_offset.sum(0, keepdim=True) - realized_offset
        ) / (self.batch_size - 1)

        with torch.no_grad():
            own = realized_offset.detach().float().flatten(1)
            loo = loo_offset.detach().float().flatten(1)
            own_norm = own.norm(dim=1)
            loo_norm = loo.norm(dim=1)
            cosine = (own * loo).sum(1) / (own_norm * loo_norm).clamp_min(1e-30)
            relative_difference = (own - loo).norm(dim=1) / own_norm.clamp_min(1e-30)
            layer = self.module_by_weight[down_weight.data_ptr()]
            for local in range(self.batch_size):
                self.geometry_rows.append({
                    "batch": self.batch,
                    "trajectory": self.batch * self.batch_size + local,
                    "t": self.t,
                    "layer": layer,
                    "own_offset_norm": float(own_norm[local].cpu()),
                    "loo_offset_norm": float(loo_norm[local].cpu()),
                    "loo_over_own_norm": float((loo_norm[local] / own_norm[local].clamp_min(1e-30)).cpu()),
                    "own_vs_loo_cosine": float(cosine[local].cpu()),
                    "relative_difference_norm": float(relative_difference[local].cpu()),
                })

        return raw_g, base + loo_offset


def loss_and_grad(model, states, ids, assistant, valid, logical_batch, iterations,
                  aux_weight):
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
            ids, mode=ForteMode.TRAIN_FIRST, embeddings=embeddings,
            embedding_mask=valid,
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


def summarize_overall(rows: list[dict], trajectories: int) -> list[dict]:
    output = []
    for objective in OBJECTIVES:
        means = {}
        for trajectory in range(trajectories):
            for scenario in SCENARIOS:
                values = [
                    row[f"{objective}_loss"] for row in rows
                    if row["trajectory"] == trajectory and row["scenario"] == scenario
                    and row["t"] > 0
                ]
                means[trajectory, scenario] = np.mean(values)
        baseline = np.asarray([means[t, SCENARIOS[0]] for t in range(trajectories)])
        ablated = np.asarray([means[t, SCENARIOS[1]] for t in range(trajectories)])
        delta = ablated - baseline
        se = delta.std(ddof=1) / math.sqrt(len(delta))
        critical = 2.131449545559323  # t(15), 97.5th percentile
        output.append({
            "objective": objective,
            "trajectories": len(delta),
            "baseline_loss": baseline.mean(),
            "loo_loss": ablated.mean(),
            "delta_loss": delta.mean(),
            "delta_percent": 100 * delta.mean() / baseline.mean(),
            "ci95_low": delta.mean() - critical * se,
            "ci95_high": delta.mean() + critical * se,
            "trajectories_worse": int((delta > 0).sum()),
        })
    return output


def summarize_by_t(rows: list[dict]) -> list[dict]:
    output = []
    for t in sorted({row["t"] for row in rows}):
        for objective in OBJECTIVES:
            baseline = np.asarray([
                row[f"{objective}_loss"] for row in rows
                if row["t"] == t and row["scenario"] == SCENARIOS[0]
            ])
            ablated = np.asarray([
                row[f"{objective}_loss"] for row in rows
                if row["t"] == t and row["scenario"] == SCENARIOS[1]
            ])
            delta = ablated - baseline
            se = delta.std(ddof=1) / math.sqrt(len(delta))
            output.append({
                "t": t,
                "objective": objective,
                "baseline_mean": baseline.mean(),
                "loo_mean": ablated.mean(),
                "delta_loss": delta.mean(),
                "delta_percent": 100 * delta.mean() / baseline.mean(),
                "ci95_low": delta.mean() - 2.131449545559323 * se,
                "ci95_high": delta.mean() + 2.131449545559323 * se,
            })
    return output


def summarize_geometry(rows: list[dict]) -> list[dict]:
    output = []
    for field in ("loo_over_own_norm", "own_vs_loo_cosine", "relative_difference_norm"):
        values = np.asarray([row[field] for row in rows])
        output.append({
            "metric": field,
            "observations": len(values),
            "mean": values.mean(),
            "median": np.median(values),
            "p25": np.quantile(values, .25),
            "p75": np.quantile(values, .75),
        })
    return output


def summarize_ranges(rows: list[dict]) -> list[dict]:
    output = []
    for name, first, last in (("early_1_8", 1, 8),
                              ("middle_9_31", 9, 31),
                              ("late_32_63", 32, 63)):
        for objective in OBJECTIVES:
            values = {}
            for scenario in SCENARIOS:
                values[scenario] = np.asarray([
                    np.mean([
                        row[f"{objective}_loss"] for row in rows
                        if row["trajectory"] == trajectory
                        and row["scenario"] == scenario
                        and first <= row["t"] <= last
                    ])
                    for trajectory in range(16)
                ])
            delta = values[SCENARIOS[1]] - values[SCENARIOS[0]]
            output.append({
                "range": name,
                "objective": objective,
                "baseline_loss": values[SCENARIOS[0]].mean(),
                "loo_loss": values[SCENARIOS[1]].mean(),
                "delta_loss": delta.mean(),
                "delta_percent": 100 * delta.mean() / values[SCENARIOS[0]].mean(),
                "trajectories_worse": int((delta > 0).sum()),
            })
    return output


def make_plot(by_t: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             constrained_layout=True)
    colors = {SCENARIOS[0]: "#4472c4", SCENARIOS[1]: "#c00000"}
    for objective, linestyle in (("assistant", "-"), ("nonassistant", "--")):
        selected = [row for row in by_t if row["objective"] == objective]
        ts = np.asarray([row["t"] for row in selected])
        axes[0].plot(ts, [row["baseline_mean"] for row in selected],
                     color=colors[SCENARIOS[0]], ls=linestyle,
                     label=f"baseline, {objective}")
        axes[0].plot(ts, [row["loo_mean"] for row in selected],
                     color=colors[SCENARIOS[1]], ls=linestyle,
                     label=f"LOO offset, {objective}")
    assistant = [row for row in by_t if row["objective"] == "assistant"]
    ts = np.asarray([row["t"] for row in assistant])
    delta = np.asarray([row["delta_percent"] for row in assistant])
    low = np.asarray([
        100 * row["ci95_low"] / row["baseline_mean"] for row in assistant
    ])
    high = np.asarray([
        100 * row["ci95_high"] / row["baseline_mean"] for row in assistant
    ])
    axes[1].fill_between(ts, low, high, color=colors[SCENARIOS[1]], alpha=.2,
                         label="paired 95% CI")
    axes[1].plot(ts, delta, color=colors[SCENARIOS[1]],
                 label="assistant paired delta")
    axes[1].axhline(0, color="black", lw=.8)
    axes[0].set(ylabel="mean loss",
                title="Replace each realized offset with the other batch members' mean")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set(xlabel="t", ylabel="assistant delta loss (%)")
    axes[1].legend()
    fig.savefig(output, dpi=180)
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
    parser.add_argument("--output-dir", type=Path, default=SRC / "local_data/"
                        "horizon_v2_forte_delta_step1000_offset_loo_batch_mean")
    parser.add_argument("--aux-loss-weight", type=float, default=.1)
    parser.add_argument("--num-logit-iterations", type=int, default=4)
    args = parser.parse_args()
    if args.trajectories % args.batch_size:
        raise ValueError("trajectories must be divisible by batch size")
    assert torch.cuda.is_available()
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
    rule = LeaveOneOutOffsetRule(model, args.batch_size)
    rule.install()
    loss_rows = []
    try:
        for start in range(0, args.trajectories, args.batch_size):
            batch = collator(raw[start:start + args.batch_size])
            ids, assistant, valid = (
                batch[key].cuda()
                for key in ("input_ids", "assistant_mask", "attention_mask")
            )
            for scenario in SCENARIOS:
                rule.enabled = scenario == SCENARIOS[1]
                rule.batch = start // args.batch_size
                model.init_state(args.batch_size, torch.device("cuda"))
                for t in range(args.chunks):
                    rule.t = t
                    al, nl = recurrent_step(
                        model, ids[:, t], assistant[:, t], valid[:, t],
                        args.batch_size, args.num_logit_iterations,
                        args.aux_loss_weight,
                    )
                    for local in range(args.batch_size):
                        loss_rows.append({
                            "trajectory": start + local,
                            "batch": start // args.batch_size,
                            "t": t,
                            "scenario": scenario,
                            "assistant_loss": float(al[local]),
                            "nonassistant_loss": float(nl[local]),
                            "weighted_total_loss": float(
                                al[local] + args.aux_loss_weight * nl[local]
                            ),
                        })
                    print(
                        f"batch {start // args.batch_size + 1}/"
                        f"{args.trajectories // args.batch_size} {scenario} "
                        f"t {t + 1}/{args.chunks}", flush=True,
                    )
                torch.cuda.empty_cache()
    finally:
        rule.remove()

    overall = summarize_overall(loss_rows, args.trajectories)
    by_t = summarize_by_t(loss_rows)
    geometry = summarize_geometry(rule.geometry_rows)
    ranges = summarize_ranges(loss_rows)
    write_csv(args.output_dir / "losses.csv", loss_rows)
    write_csv(args.output_dir / "overall_summary.csv", overall)
    write_csv(args.output_dir / "losses_by_t.csv", by_t)
    write_csv(args.output_dir / "offset_loo_geometry.csv", rule.geometry_rows)
    write_csv(args.output_dir / "offset_loo_geometry_summary.csv", geometry)
    write_csv(args.output_dir / "losses_by_t_range.csv", ranges)
    make_plot(by_t, args.output_dir / "offset_loo_losses_through_t.png")
    metadata = {
        "checkpoint": args.checkpoint,
        "step": args.step,
        "dataset": config.dataset.url,
        "trajectories": args.trajectories,
        "batch_size": args.batch_size,
        "chunks": args.chunks,
        "intervention": "at each inner step and layer, replace each trajectory's realized norm-capped offset update matrix with the arithmetic mean of the other batch members' realized offset matrices; retain its own no-offset update",
        "leave_one_out_members": args.batch_size - 1,
        "loss_chunks_summarized": "1..63 (t=0 is pre-update and identical)",
        "aux_loss_weight": args.aux_loss_weight,
        "num_logit_iterations": args.num_logit_iterations,
        "torch_xla_used": False,
        "cuda_device": torch.cuda.get_device_name(),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"overall": overall, "geometry": geometry}, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
