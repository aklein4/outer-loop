from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn.functional as F

from data.datasets import get_dataset
from models import load_checkpoint
from models.slither import SlitherStateMechanism
from utils import constants
from utils.import_utils import import_collator


STATE_RE = re.compile(
    r"^(backbone|output|memory)_layers\.layers\.(\d+)\.state_mechanism$"
)
ATTN_RE = re.compile(
    r"^(backbone|output|memory)_layers\.layers\.(\d+)\.self_attn$"
)
WINDOWS = (1, 8, 32, 64, 128, 256, 1024)
POSITION_BINS = (
    (0, 1, "0"),
    (1, 2, "1"),
    (2, 4, "2-3"),
    (4, 8, "4-7"),
    (8, 16, "8-15"),
    (16, 32, "16-31"),
    (32, 64, "32-63"),
    (64, 128, "64-127"),
    (128, 256, "128-255"),
    (256, 512, "256-511"),
    (512, 1024, "512-1023"),
)
PROBE_POSITIONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1023)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="aklein4/slither_mesa-v2-350m")
    parser.add_argument("--checkpoint-step", type=int, default=250)
    parser.add_argument("--data-config", default="data/longattn-smollm2.yaml")
    parser.add_argument("--trainer-config", default="trainer/slither-med.yaml")
    parser.add_argument("--num-loss-examples", type=int, default=32)
    parser.add_argument("--num-gate-examples", type=int, default=4)
    parser.add_argument("--num-ablation-examples", type=int, default=8)
    parser.add_argument(
        "--selected-chunks", type=int, nargs="+", default=[1, 8, 16, 24, 30]
    )
    parser.add_argument("--attention-probe-chunk", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=constants.LOCAL_DATA_PATH / "slither_mesa_v2_analysis",
    )
    return parser.parse_args()


def load_tokens(data_config, count):
    dataset = get_dataset(data_config.dataset.url, data_config.dataset.kwargs)
    iterator = iter(dataset)
    rows = [next(iterator) for _ in range(count)]
    collator = import_collator(data_config.collator.type)(
        **data_config.collator.kwargs
    )
    tokens = collator(rows)["input_ids"]
    del iterator, dataset, rows
    gc.collect()
    return tokens


def module_metadata(model):
    states = []
    attentions = []
    for name, module in model.named_modules():
        match = STATE_RE.match(name)
        if match is not None:
            states.append(
                {
                    "family": match.group(1),
                    "layer": int(match.group(2)),
                    "module": module,
                }
            )
        match = ATTN_RE.match(name)
        if match is not None:
            attentions.append(
                {
                    "family": match.group(1),
                    "layer": int(match.group(2)),
                    "module": module,
                }
            )
    if len(states) != 24 or len(attentions) != 24:
        raise RuntimeError(
            f"Expected 24 state and attention modules, got "
            f"{len(states)} and {len(attentions)}"
        )
    return states, attentions


class GateCollector:
    def __init__(self):
        self.enabled = False
        self.chunk = -1
        self.stats = {}
        self.samples = defaultdict(list)
        self.position_rows = []

    @staticmethod
    def _thresholds(kind, heads):
        if kind.endswith("_out") or kind == "attention":
            return 0.1, 1.9
        return 0.1, heads / 2

    def add(self, kind, family, layer, values):
        if not self.enabled:
            return
        values = values.detach().float()
        batch, length, heads = values.shape
        low_threshold, high_threshold = self._thresholds(kind, heads)
        flat = values.reshape(-1, heads)
        key = (kind, family, layer)
        sums = flat.sum(0).double().cpu().numpy()
        squares = flat.square().sum(0).double().cpu().numpy()
        lows = (flat < low_threshold).sum(0).double().cpu().numpy()
        highs = (flat > high_threshold).sum(0).double().cpu().numpy()
        minima = flat.amin(0).cpu().numpy()
        maxima = flat.amax(0).cpu().numpy()
        if key not in self.stats:
            self.stats[key] = {
                "count": 0,
                "sum": np.zeros(heads),
                "sum2": np.zeros(heads),
                "low": np.zeros(heads),
                "high": np.zeros(heads),
                "min": np.full(heads, np.inf),
                "max": np.full(heads, -np.inf),
            }
        state = self.stats[key]
        state["count"] += flat.shape[0]
        state["sum"] += sums
        state["sum2"] += squares
        state["low"] += lows
        state["high"] += highs
        state["min"] = np.minimum(state["min"], minima)
        state["max"] = np.maximum(state["max"], maxima)

        sampled = values[:, ::32].permute(2, 0, 1).reshape(heads, -1).cpu()
        for head in range(heads):
            self.samples[(kind, family, layer, head)].append(
                sampled[head].numpy()
            )

        for start, stop, label in POSITION_BINS:
            if start >= length:
                continue
            part = values[:, start : min(stop, length)]
            if kind.endswith("_in"):
                probabilities = part / heads
                routing_entropy = -(
                    probabilities.clamp_min(1e-12)
                    * probabilities.clamp_min(1e-12).log()
                ).sum(dim=-1) / math.log(heads)
                normalized_entropy = float(routing_entropy.mean())
                peak_gate = float(part.amax(dim=-1).mean())
            else:
                normalized_entropy = math.nan
                peak_gate = math.nan
            self.position_rows.append(
                {
                    "kind": kind,
                    "family": family,
                    "layer": layer,
                    "chunk": self.chunk,
                    "position_bin": label,
                    "position_start": start,
                    "count": int(part.numel()),
                    "mean": float(part.mean()),
                    "std": float(part.std(unbiased=False)),
                    "low_fraction": float((part < low_threshold).float().mean()),
                    "high_fraction": float((part > high_threshold).float().mean()),
                    "normalized_routing_entropy": normalized_entropy,
                    "mean_peak_gate": peak_gate,
                }
            )

    def frames(self):
        head_rows = []
        sample_rows = []
        for (kind, family, layer), state in self.stats.items():
            count = state["count"]
            mean = state["sum"] / count
            variance = np.maximum(state["sum2"] / count - mean**2, 0)
            for head in range(len(mean)):
                arrays = self.samples[(kind, family, layer, head)]
                sample = np.concatenate(arrays) if arrays else np.empty(0)
                quantiles = (
                    np.quantile(sample, [0.01, 0.05, 0.5, 0.95, 0.99])
                    if sample.size
                    else np.full(5, np.nan)
                )
                head_rows.append(
                    {
                        "kind": kind,
                        "family": family,
                        "layer": layer,
                        "head": head,
                        "count": count,
                        "mean": mean[head],
                        "std": math.sqrt(variance[head]),
                        "min": state["min"][head],
                        "max": state["max"][head],
                        "low_fraction": state["low"][head] / count,
                        "high_fraction": state["high"][head] / count,
                        "q01": quantiles[0],
                        "q05": quantiles[1],
                        "median": quantiles[2],
                        "q95": quantiles[3],
                        "q99": quantiles[4],
                    }
                )
                for value in sample:
                    sample_rows.append(
                        {
                            "kind": kind,
                            "family": family,
                            "layer": layer,
                            "head": head,
                            "value": float(value),
                        }
                    )
        return (
            pd.DataFrame(head_rows),
            pd.DataFrame(self.position_rows),
            pd.DataFrame(sample_rows),
        )


def register_gate_hooks(states, attentions, collector):
    handles = []
    for metadata in attentions:
        module = metadata["module"]

        def hook(_module, _inputs, output, *, metadata=metadata):
            collector.add(
                "attention",
                metadata["family"],
                metadata["layer"],
                2.0 * torch.sigmoid(output.float()),
            )

        handles.append(module.gate_proj.register_forward_hook(hook))

    for metadata in states:
        mechanism = metadata["module"]
        for kind, linear, heads, activation in (
            (
                "read_in",
                mechanism.in_gate,
                mechanism.num_state_in_heads,
                "softmax",
            ),
            (
                "read_out",
                mechanism.out_gate,
                mechanism.num_state_out_heads,
                "sigmoid",
            ),
            (
                "write_in",
                mechanism.writer.in_gate,
                mechanism.num_state_in_heads,
                "softmax",
            ),
            (
                "write_out",
                mechanism.writer.out_gate,
                mechanism.num_state_out_heads,
                "sigmoid",
            ),
        ):

            def hook(
                _module,
                _inputs,
                output,
                *,
                metadata=metadata,
                kind=kind,
                heads=heads,
                activation=activation,
            ):
                if activation == "softmax":
                    values = torch.softmax(output.float(), -1) * heads
                else:
                    values = 2.0 * torch.sigmoid(output.float())
                collector.add(
                    kind, metadata["family"], metadata["layer"], values
                )

            handles.append(linear.register_forward_hook(hook))
    return handles


def register_attention_stream_hooks(attentions, context, records):
    handles = []
    pending_gates = {}
    for metadata in attentions:
        if metadata["family"] == "memory":
            continue
        module = metadata["module"]
        key = (metadata["family"], metadata["layer"])

        def attention_pre(
            _module, args, kwargs, *, metadata=metadata, key=key
        ):
            if not context["capture_attention"]:
                return
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            pending_gates[key] = (
                2.0
                * torch.sigmoid(
                    F.linear(hidden, _module.gate_proj.weight).float()
                )
            ).detach()

        def block_pre(_module, args, *, metadata=metadata, key=key):
            if not context["capture_attention"]:
                return
            query, keys = args[:2]
            batch, heads, full_length, head_dim = query.shape
            memory_length = context["memory_length"]
            if keys.shape[1] != heads:
                keys = keys.repeat_interleave(heads // keys.shape[1], dim=1)
            gates = pending_gates.pop(key)
            for position in PROBE_POSITIONS:
                query_index = memory_length + position
                if query_index >= full_length:
                    continue
                allowed = query_index + 1
                q = query[:, :, query_index].float()
                k = keys[:, :, :allowed].float()
                scores = torch.einsum("bhd,bhkd->bhk", q, k) / math.sqrt(
                    head_dim
                )
                weights = torch.softmax(scores, -1)
                memory_mass = weights[:, :, :memory_length].sum(-1)
                self_weight = weights[:, :, -1]
                for example in range(batch):
                    for head in range(heads):
                        records.append(
                            {
                                "example": example,
                                "family": metadata["family"],
                                "layer": metadata["layer"],
                                "head": head,
                                "position": position,
                                "gate": float(gates[example, position, head]),
                                "memory_mass": float(
                                    memory_mass[example, head]
                                ),
                                "self_weight": float(
                                    self_weight[example, head]
                                ),
                            }
                        )

        handles.append(
            module.register_forward_pre_hook(attention_pre, with_kwargs=True)
        )
        handles.append(
            module.attention_block.register_forward_pre_hook(block_pre)
        )
    return handles


@torch.inference_mode()
def run_gate_pass(
    model,
    tokens,
    use_autocast,
    states,
    attentions,
    probe_chunk,
):
    device = torch.device("cuda")
    tokens = tokens.to(device)
    inputs = list(tokens[:, :-1].split(model.chunk_length, 1))
    model.init_state(tokens.shape[0], device)
    model.empty_state()
    previous_mem = None
    collector = GateCollector()
    collector.enabled = True
    stream_records = []
    context = {
        "capture_attention": False,
        "memory_length": model.chunk_length,
    }
    handles = register_gate_hooks(states, attentions, collector)
    handles += register_attention_stream_hooks(
        attentions, context, stream_records
    )
    try:
        for chunk, input_ids in enumerate(inputs):
            collector.chunk = chunk
            context["capture_attention"] = (
                chunk == probe_chunk and previous_mem is not None
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                _, new_mem = model(input_ids, mem_states=previous_mem)
            new_mem = new_mem.float()
            context["capture_attention"] = False
            if chunk < len(inputs) - 1:
                model.increment_state(new_mem)
            previous_mem = new_mem
            print(f"gate pass chunk {chunk:02d}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        model.empty_state()
    return (*collector.frames(), pd.DataFrame(stream_records))


@torch.inference_mode()
def run_loss_pass(model, tokens, use_autocast):
    device = torch.device("cuda")
    tokens = tokens.to(device)
    inputs = list(tokens[:, :-1].split(model.chunk_length, 1))
    labels = list(tokens[:, 1:].split(model.chunk_length, 1))
    model.init_state(tokens.shape[0], device)
    model.empty_state()
    previous_mem = None
    all_losses = []
    for chunk, (input_ids, targets) in enumerate(zip(inputs, labels)):
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            logits, new_mem = model(input_ids, mem_states=previous_mem)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        all_losses.append(losses.float().cpu())
        new_mem = new_mem.float()
        if chunk < len(inputs) - 1:
            model.increment_state(new_mem)
        previous_mem = new_mem
        print(f"loss pass chunk {chunk:02d}", flush=True)
    model.empty_state()
    return torch.cat(all_losses, 1).numpy()


def summarize_losses(losses, chunk_length):
    num_examples, total = losses.shape
    boundary_rows = []
    for width in WINDOWS:
        if width > chunk_length:
            continue
        before = []
        after = []
        for boundary in range(chunk_length, total, chunk_length):
            if boundary + width > total:
                continue
            before.append(losses[:, boundary - width : boundary].mean(1))
            after.append(losses[:, boundary : boundary + width].mean(1))
        before = np.concatenate(before)
        after = np.concatenate(after)
        delta = after - before
        boundary_rows.append(
            {
                "window_tokens": width,
                "before_mean": before.mean(),
                "after_mean": after.mean(),
                "delta": delta.mean(),
                "relative_delta": delta.mean() / before.mean(),
                "delta_sem": delta.std(ddof=1) / math.sqrt(len(delta)),
                "positive_fraction": (delta > 0).mean(),
            }
        )

    position_rows = []
    chunks = [
        losses[:, start : min(start + chunk_length, total)]
        for start in range(0, total, chunk_length)
    ]
    later = chunks[1:]
    for position in range(chunk_length):
        values = [chunk[:, position] for chunk in later if position < chunk.shape[1]]
        values = np.concatenate(values)
        position_rows.append(
            {
                "position": position,
                "mean_loss": values.mean(),
                "std": values.std(),
                "count": len(values),
            }
        )

    offset_rows = []
    for offset in range(-256, 257):
        values = []
        for boundary in range(chunk_length, total, chunk_length):
            index = boundary + offset
            if 0 <= index < total:
                values.append(losses[:, index])
        values = np.concatenate(values)
        offset_rows.append(
            {
                "offset": offset,
                "mean_loss": values.mean(),
                "std": values.std(),
                "count": len(values),
            }
        )

    chunk_rows = []
    for chunk, values in enumerate(chunks):
        chunk_rows.append(
            {
                "chunk": chunk,
                "mean_loss": values.mean(),
                "first_token_loss": values[:, 0].mean(),
            }
        )
    return (
        pd.DataFrame(boundary_rows),
        pd.DataFrame(position_rows),
        pd.DataFrame(offset_rows),
        pd.DataFrame(chunk_rows),
    )


def state_scale_matches(metadata, condition):
    if condition == "baseline":
        return False
    if condition in ("no_all_state", "no_causal_state"):
        return (
            True
            if condition == "no_all_state"
            else metadata["family"] in ("backbone", "output")
        )
    if condition == "no_backbone_state":
        return metadata["family"] == "backbone"
    if condition == "no_output_state":
        return metadata["family"] == "output"
    if condition == "no_memory_layer_state":
        return metadata["family"] == "memory"
    return False


def register_state_ablation_hooks(states, context):
    handles = []
    for metadata in states:
        module = metadata["module"]

        def hook(_module, _inputs, output, *, metadata=metadata):
            if state_scale_matches(metadata, context["condition"]):
                return torch.zeros_like(output)
            return None

        handles.append(module.register_forward_hook(hook))
    return handles


def loss_windows(losses, condition, chunk, records):
    for width in WINDOWS:
        width = min(width, losses.shape[1])
        values = losses[:, :width].mean(1).detach().float().cpu().numpy()
        for example, value in enumerate(values):
            records.append(
                {
                    "condition": condition,
                    "example": example,
                    "chunk": chunk,
                    "window_tokens": width,
                    "loss": value,
                }
            )


def shuffle_state_batch(states, shift):
    with torch.no_grad():
        for metadata in states:
            module = metadata["module"]
            module.state.copy_(module.state.roll(shift, 0))
            module.k_corr.copy_(module.k_corr.roll(shift, 0))
            module.k_count.copy_(module.k_count.roll(shift, 0))


@torch.inference_mode()
def run_pointwise_ablations(
    model, tokens, use_autocast, states, selected_chunks
):
    device = torch.device("cuda")
    tokens = tokens.to(device)
    inputs = list(tokens[:, :-1].split(model.chunk_length, 1))
    labels = list(tokens[:, 1:].split(model.chunk_length, 1))
    model.init_state(tokens.shape[0], device)
    model.empty_state()
    previous_mem = None
    records = []
    context = {"condition": "baseline"}
    handles = register_state_ablation_hooks(states, context)

    def forward(input_ids, targets, mem):
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            logits, new_mem = model(input_ids, mem_states=mem)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        return losses.float(), new_mem.float()

    try:
        for chunk, (input_ids, targets) in enumerate(zip(inputs, labels)):
            context["condition"] = "baseline"
            baseline_losses, new_mem = forward(
                input_ids, targets, previous_mem
            )
            if chunk in selected_chunks:
                loss_windows(baseline_losses, "baseline", chunk, records)
                for condition in (
                    "no_causal_state",
                    "no_backbone_state",
                    "no_output_state",
                ):
                    context["condition"] = condition
                    losses, _ = forward(input_ids, targets, previous_mem)
                    loss_windows(losses, condition, chunk, records)

                context["condition"] = "baseline"
                shuffle_state_batch(states, 1)
                losses, _ = forward(input_ids, targets, previous_mem)
                shuffle_state_batch(states, -1)
                loss_windows(
                    losses, "cross_example_shuffled_state", chunk, records
                )

                empty_mem = previous_mem[:, :0]
                losses, _ = forward(input_ids, targets, empty_mem)
                loss_windows(losses, "state_only_no_memory_tokens", chunk, records)
                context["condition"] = "no_causal_state"
                losses, _ = forward(input_ids, targets, empty_mem)
                loss_windows(losses, "neither_state_nor_memory", chunk, records)
                context["condition"] = "baseline"

            if chunk < len(inputs) - 1:
                model.increment_state(new_mem)
            previous_mem = new_mem
            print(f"pointwise ablation chunk {chunk:02d}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        model.empty_state()
    return pd.DataFrame(records)


@torch.inference_mode()
def run_rollout_ablations(
    model, tokens, use_autocast, states, selected_chunks
):
    device = torch.device("cuda")
    tokens = tokens.to(device)
    inputs = list(tokens[:, :-1].split(model.chunk_length, 1))
    labels = list(tokens[:, 1:].split(model.chunk_length, 1))
    records = []
    context = {"condition": "baseline"}
    handles = register_state_ablation_hooks(states, context)
    conditions = (
        "baseline",
        "no_all_state",
        "no_causal_state",
        "no_backbone_state",
        "no_output_state",
        "no_memory_layer_state",
    )
    try:
        for condition in conditions:
            context["condition"] = condition
            model.init_state(tokens.shape[0], device)
            model.empty_state()
            previous_mem = None
            for chunk, (input_ids, targets) in enumerate(zip(inputs, labels)):
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=use_autocast
                ):
                    logits, new_mem = model(input_ids, mem_states=previous_mem)
                if chunk in selected_chunks:
                    losses = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="none",
                    ).view_as(targets)
                    loss_windows(losses, condition, chunk, records)
                new_mem = new_mem.float()
                if chunk < len(inputs) - 1:
                    model.increment_state(new_mem)
                previous_mem = new_mem
            model.empty_state()
            print(f"rollout {condition} complete", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return pd.DataFrame(records)


def paired_summary(records, baseline="baseline"):
    keys = ["example", "chunk", "window_tokens"]
    base = records[records.condition == baseline][keys + ["loss"]].rename(
        columns={"loss": "baseline_loss"}
    )
    merged = records.merge(base, on=keys)
    rows = []
    for (condition, width), frame in merged.groupby(
        ["condition", "window_tokens"]
    ):
        delta = frame.loss - frame.baseline_loss
        rows.append(
            {
                "condition": condition,
                "window_tokens": width,
                "mean_loss": frame.loss.mean(),
                "baseline_loss": frame.baseline_loss.mean(),
                "delta": delta.mean(),
                "delta_sem": delta.std(ddof=1) / math.sqrt(len(delta)),
                "fraction_worse": (delta > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_gate_distributions(samples, output):
    state_kinds = ("read_in", "read_out", "write_in", "write_out")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, kind in zip(axes.flat, state_kinds):
        frame = samples[samples.kind == kind]
        for family in ("backbone", "output", "memory"):
            values = frame[frame.family == family].value
            ax.hist(
                values,
                bins=80,
                range=(0, 4 if kind.endswith("_in") else 2),
                density=True,
                histtype="step",
                linewidth=1.5,
                label=family,
            )
        ax.axvline(1, color="black", alpha=0.3)
        ax.set(title=kind.replace("_", " "), xlabel="gate multiplier", ylabel="density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_attention_gates(samples, output):
    frame = samples[samples.kind == "attention"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for family in ("backbone", "output", "memory"):
        values = frame[frame.family == family].value
        axes[0].hist(
            values,
            bins=80,
            range=(0, 2),
            density=True,
            histtype="step",
            linewidth=1.5,
            label=family,
        )
    axes[0].axvline(1, color="black", alpha=0.3)
    axes[0].set(xlabel="attention-head multiplier", ylabel="density", title="Distribution")
    axes[0].legend()

    means = (
        frame.groupby(["family", "layer"], as_index=False).value.mean()
    )
    for family, values in means.groupby("family"):
        offset = {"backbone": 0, "output": 16, "memory": 20}[family]
        axes[1].plot(
            values.layer + offset,
            values.value,
            marker="o",
            label=family,
        )
    axes[1].axhline(1, color="black", alpha=0.3)
    axes[1].set(
        xlabel="global layer index",
        ylabel="mean gate multiplier",
        title="Layer means",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_loss_boundary(offsets, positions, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(offsets.offset, offsets.mean_loss)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set(
        xlabel="offset from chunk boundary",
        ylabel="mean loss",
        title="Boundary-aligned loss",
    )
    axes[0].grid(alpha=0.2)
    axes[1].plot(positions.position, positions.mean_loss)
    axes[1].set(
        xlabel="within-chunk position",
        ylabel="mean loss",
        title="Loss by within-chunk position",
    )
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_state_ablations(pointwise, rollout, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, frame, title in (
        (axes[0], pointwise, "Pointwise pathway ablation"),
        (axes[1], rollout, "Persistent rollout ablation"),
    ):
        values = frame[frame.window_tokens <= 256]
        for condition, group in values.groupby("condition"):
            ax.plot(
                group.window_tokens,
                group.delta,
                marker="o",
                label=condition.replace("_", " "),
            )
        ax.set_xscale("log", base=2)
        ax.axhline(0, color="black", linewidth=1)
        ax.set(xlabel="prefix window", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("loss delta versus baseline")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_attention_patterns(stream, output):
    means = stream.groupby(["family", "position"], as_index=False).agg(
        gate=("gate", "mean"),
        memory_mass=("memory_mass", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for family, values in means.groupby("family"):
        axes[0].plot(values.position, values.gate, marker="o", label=family)
        axes[1].plot(
            values.position, values.memory_mass, marker="o", label=family
        )
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set(
        xlabel="within-chunk position",
        ylabel="mean gate",
        title="Attention gate by position",
    )
    axes[1].set(
        xlabel="within-chunk position",
        ylabel="memory attention mass",
        title="Encoded-memory allocation",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data_config = OmegaConf.load(constants.CONFIG_PATH(args.data_config))
    trainer_config = OmegaConf.load(constants.CONFIG_PATH(args.trainer_config))
    output = args.output_dir / (
        f"{args.checkpoint.replace('/', '--')}_step={args.checkpoint_step}"
    )
    output.mkdir(parents=True, exist_ok=True)
    max_examples = max(
        args.num_loss_examples,
        args.num_gate_examples,
        args.num_ablation_examples,
    )
    tokens = load_tokens(data_config, max_examples)
    model = load_checkpoint(
        args.checkpoint,
        args.checkpoint_step,
        attention_kernel="gpu_flash_attention",
    )
    model.to("cuda", dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    states, attentions = module_metadata(model)
    use_autocast = bool(trainer_config.use_autocast)
    started = time.time()

    losses = run_loss_pass(
        model, tokens[: args.num_loss_examples], use_autocast
    )
    boundary, positions, offsets, chunks = summarize_losses(
        losses, model.chunk_length
    )
    boundary.to_csv(output / "boundary_windows.csv", index=False)
    positions.to_csv(output / "loss_by_position.csv", index=False)
    offsets.to_csv(output / "loss_around_boundaries.csv", index=False)
    chunks.to_csv(output / "loss_by_chunk.csv", index=False)
    np.save(output / "token_losses.npy", losses)
    plot_loss_boundary(offsets, positions, output / "boundary_loss.png")

    heads, gate_positions, samples, stream = run_gate_pass(
        model,
        tokens[: args.num_gate_examples],
        use_autocast,
        states,
        attentions,
        args.attention_probe_chunk,
    )
    heads.to_csv(output / "gate_head_stats.csv", index=False)
    gate_positions.to_csv(output / "gate_position_stats.csv", index=False)
    samples.to_csv(output / "gate_samples.csv", index=False)
    stream.to_csv(output / "attention_stream_patterns.csv", index=False)
    plot_gate_distributions(samples, output / "state_gate_distributions.png")
    plot_attention_gates(samples, output / "attention_gate_distributions.png")
    plot_attention_patterns(stream, output / "attention_patterns.png")

    pointwise_records = run_pointwise_ablations(
        model,
        tokens[: args.num_ablation_examples],
        use_autocast,
        states,
        set(args.selected_chunks),
    )
    rollout_records = run_rollout_ablations(
        model,
        tokens[: args.num_ablation_examples],
        use_autocast,
        states,
        set(args.selected_chunks),
    )
    pointwise = paired_summary(pointwise_records)
    rollout = paired_summary(rollout_records)
    pointwise_records.to_csv(output / "pointwise_ablation_records.csv", index=False)
    rollout_records.to_csv(output / "rollout_ablation_records.csv", index=False)
    pointwise.to_csv(output / "pointwise_ablation_summary.csv", index=False)
    rollout.to_csv(output / "rollout_ablation_summary.csv", index=False)
    plot_state_ablations(pointwise, rollout, output / "state_ablations.png")

    family_gate = (
        heads.groupby(["kind", "family"], as_index=False)
        .agg(
            mean=("mean", "mean"),
            head_std_median=("std", "median"),
            q01=("q01", "mean"),
            median=("median", "mean"),
            q99=("q99", "mean"),
            low_fraction=("low_fraction", "mean"),
            high_fraction=("high_fraction", "mean"),
        )
    )
    stream_summary = (
        stream.groupby(["family", "position"], as_index=False)
        .agg(
            gate=("gate", "mean"),
            memory_mass=("memory_mass", "mean"),
            self_weight=("self_weight", "mean"),
        )
    )
    family_gate.to_csv(output / "gate_family_summary.csv", index=False)
    stream_summary.to_csv(output / "attention_stream_summary.csv", index=False)
    machine = {
        "checkpoint": args.checkpoint,
        "step": args.checkpoint_step,
        "dataset": data_config.dataset.url,
        "num_loss_examples": args.num_loss_examples,
        "num_gate_examples": args.num_gate_examples,
        "num_ablation_examples": args.num_ablation_examples,
        "elapsed_seconds": time.time() - started,
        "overall_loss": float(losses.mean()),
        "boundary": boundary.to_dict(orient="records"),
        "gate_family": family_gate.to_dict(orient="records"),
        "attention_stream": stream_summary.to_dict(orient="records"),
        "pointwise_ablation": pointwise.to_dict(orient="records"),
        "rollout_ablation": rollout.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(machine, indent=2) + "\n")
    print(json.dumps(machine, indent=2), flush=True)
    print(f"Wrote analysis to {output}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
