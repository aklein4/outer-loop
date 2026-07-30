from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time

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
COMPONENT_RE = re.compile(
    r"^(backbone|output|memory)_layers\.layers\.(\d+)\.(self_attn|mlp)$"
)
WINDOWS = (1, 8, 32, 128, 256, 1024)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure direct and rollout-level reliance on Slither matrix state."
    )
    parser.add_argument("--checkpoint", default="aklein4/slither_alpha-350m")
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument("--data-config", default="data/longattn-smollm2.yaml")
    parser.add_argument("--trainer-config", default="trainer/slither-med.yaml")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument(
        "--selected-chunks", type=int, nargs="+", default=[1, 8, 16, 24, 30]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=constants.LOCAL_DATA_PATH / "slither_state_reliance",
    )
    return parser.parse_args()


def load_tokens(data_config, num_examples):
    dataset = get_dataset(data_config.dataset.url, data_config.dataset.kwargs)
    iterator = iter(dataset)
    rows = [next(iterator) for _ in range(num_examples)]
    collator = import_collator(data_config.collator.type)(
        **data_config.collator.kwargs
    )
    tokens = collator(rows)["input_ids"]
    del iterator, dataset, rows
    gc.collect()
    return tokens


def state_modules(model):
    result = []
    for name, module in model.named_modules():
        match = STATE_RE.match(name)
        if match is not None:
            result.append(
                {
                    "name": name,
                    "family": match.group(1),
                    "layer": int(match.group(2)),
                    "module": module,
                }
            )
    if len(result) != 24:
        raise RuntimeError(f"Expected 24 state mechanisms, got {len(result)}")
    return result


def matches_condition(metadata, condition):
    kind = condition["kind"]
    if kind == "identity":
        return True
    if kind == "family":
        return metadata["family"] == condition["family"]
    if kind == "module":
        return (
            metadata["family"] == condition["family"]
            and metadata["layer"] == condition["layer"]
        )
    if kind == "causal":
        return metadata["family"] in ("backbone", "output")
    raise ValueError(kind)


def register_state_hooks(modules, context, activation_records):
    handles = []
    for metadata in modules:
        module = metadata["module"]

        def hook(_module, inputs, kwargs, output, *, metadata=metadata):
            if context["capture_activations"]:
                x = (
                    inputs[0] if inputs else kwargs["hidden_states"]
                ).detach().float()
                y = output.detach().float()
                for width in WINDOWS:
                    width = min(width, y.shape[1])
                    x_rms = x[:, :width].square().mean((-2, -1)).sqrt()
                    y_rms = y[:, :width].square().mean((-2, -1)).sqrt()
                    for example in range(y.shape[0]):
                        activation_records.append(
                            {
                                "example": example,
                                "chunk": context["chunk"],
                                "family": metadata["family"],
                                "layer": metadata["layer"],
                                "window_tokens": width,
                                "normalized_input_rms": float(x_rms[example]),
                                "state_output_rms": float(y_rms[example]),
                                "out_scale": float(_module.get_out_scale()),
                            }
                        )
            scale = 1.0
            for condition, value in context["scales"]:
                if matches_condition(metadata, condition):
                    scale *= value
            if scale != 1.0:
                return output * scale
            return None

        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    return handles


def register_component_hooks(model, context, component_records):
    handles = []
    for name, module in model.named_modules():
        match = COMPONENT_RE.match(name)
        if match is None:
            continue
        family, layer, component = match.group(1), int(match.group(2)), match.group(3)

        def hook(_module, _inputs, output, *, family=family, layer=layer, component=component):
            if not context["capture_activations"]:
                return
            y = output.detach().float()
            for width in WINDOWS:
                width = min(width, y.shape[1])
                rms = y[:, :width].square().mean((-2, -1)).sqrt()
                for example in range(y.shape[0]):
                    component_records.append(
                        {
                            "example": example,
                            "chunk": context["chunk"],
                            "family": family,
                            "layer": layer,
                            "component": component,
                            "window_tokens": width,
                            "output_rms": float(rms[example]),
                        }
                    )

        handles.append(module.register_forward_hook(hook))
    return handles


def forward_losses(model, input_ids, labels, mem_states, use_autocast):
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
    ):
        logits, new_mem = model(input_ids=input_ids, mem_states=mem_states)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    return losses.float(), new_mem.float()


def append_loss_windows(records, condition, losses, chunk):
    losses = losses.detach().float()
    for width in WINDOWS:
        width = min(width, losses.shape[1])
        means = losses[:, :width].mean(1).cpu().numpy()
        for example, value in enumerate(means):
            records.append(
                {
                    "condition": condition,
                    "example": example,
                    "chunk": chunk,
                    "window_tokens": width,
                    "loss": float(value),
                }
            )


def snapshot_states(modules):
    return [metadata["module"].state.detach().clone() for metadata in modules]


@torch.no_grad()
def restore_states(modules, values):
    for metadata, value in zip(modules, values):
        metadata["module"].state.copy_(value)


@torch.no_grad()
def writer_updates(modules, mem_states):
    return [
        metadata["module"].writer(mem_states).detach().float()
        for metadata in modules
    ]


@torch.no_grad()
def record_state_geometry(records, modules, latest_updates, chunk):
    original = snapshot_states(modules)
    effective_original = [
        metadata["module"].get_s().detach().float().clone()
        for metadata in modules
    ]
    restore_states(modules, latest_updates)
    effective_latest = [
        metadata["module"].get_s().detach().float().clone()
        for metadata in modules
    ]
    restore_states(modules, original)

    for metadata, state, update, effective, latest_effective in zip(
        modules,
        original,
        latest_updates,
        effective_original,
        effective_latest,
    ):
        for example in range(state.shape[0]):
            s = state[example].double().flatten()
            u = update[example].double().flatten()
            es = effective[example].double().flatten()
            eu = latest_effective[example].double().flatten()
            other = (example + 1) % state.shape[0]
            s_other = state[other].double().flatten()
            u_other = update[other].double().flatten()
            es_other = effective[other].double().flatten()
            records.append(
                {
                    "example": example,
                    "chunk": chunk,
                    "family": metadata["family"],
                    "layer": metadata["layer"],
                    "raw_state_rms": float(s.square().mean().sqrt()),
                    "latest_update_rms": float(u.square().mean().sqrt()),
                    "latest_to_state_norm_ratio": float(
                        u.norm() / s.norm().clamp_min(1e-30)
                    ),
                    "raw_state_latest_cosine": float(
                        torch.dot(s, u)
                        / (s.norm() * u.norm()).clamp_min(1e-30)
                    ),
                    "effective_latest_cosine": float(
                        torch.dot(es, eu)
                        / (es.norm() * eu.norm()).clamp_min(1e-30)
                    ),
                    "raw_state_cross_example_cosine": float(
                        torch.dot(s, s_other)
                        / (s.norm() * s_other.norm()).clamp_min(1e-30)
                    ),
                    "latest_update_cross_example_cosine": float(
                        torch.dot(u, u_other)
                        / (u.norm() * u_other.norm()).clamp_min(1e-30)
                    ),
                    "effective_state_cross_example_cosine": float(
                        torch.dot(es, es_other)
                        / (es.norm() * es_other.norm()).clamp_min(1e-30)
                    ),
                    "effective_state_rms": float(
                        es.square().mean().sqrt()
                    ),
                }
            )


@torch.inference_mode()
def pointwise_analysis(
    model,
    modules,
    context,
    tokens,
    selected_chunks,
    use_autocast,
):
    device = torch.device("cuda")
    tokens = tokens.to(device)
    chunk_length = int(model.chunk_length)
    input_chunks = list(tokens[:, :-1].split(chunk_length, dim=1))
    label_chunks = list(tokens[:, 1:].split(chunk_length, dim=1))
    model.init_state(tokens.shape[0], device)
    model.empty_state()
    previous_mem = None
    first_update_states = None
    loss_records = []
    geometry_records = []
    started = time.time()

    scale_conditions = {
        "state_scale_-1": [({"kind": "causal"}, -1.0)],
        "state_scale_0": [({"kind": "causal"}, 0.0)],
        "state_scale_0.5": [({"kind": "causal"}, 0.5)],
        "state_scale_2": [({"kind": "causal"}, 2.0)],
        "state_scale_4": [({"kind": "causal"}, 4.0)],
        "no_backbone_state": [({"kind": "family", "family": "backbone"}, 0.0)],
        "no_output_state": [({"kind": "family", "family": "output"}, 0.0)],
    }

    for chunk, (input_ids, labels) in enumerate(zip(input_chunks, label_chunks)):
        selected = chunk in selected_chunks
        context["chunk"] = chunk
        context["scales"] = []
        context["capture_activations"] = selected
        baseline_loss, new_mem = forward_losses(
            model, input_ids, labels, previous_mem, use_autocast
        )
        context["capture_activations"] = False

        if selected:
            append_loss_windows(loss_records, "baseline", baseline_loss, chunk)
            for name, scales in scale_conditions.items():
                context["scales"] = scales
                losses, _ = forward_losses(
                    model, input_ids, labels, previous_mem, use_autocast
                )
                append_loss_windows(loss_records, name, losses, chunk)
            context["scales"] = []

            for metadata in modules:
                if metadata["family"] not in ("backbone", "output"):
                    continue
                context["scales"] = [
                    (
                        {
                            "kind": "module",
                            "family": metadata["family"],
                            "layer": metadata["layer"],
                        },
                        0.0,
                    )
                ]
                losses, _ = forward_losses(
                    model, input_ids, labels, previous_mem, use_autocast
                )
                append_loss_windows(
                    loss_records,
                    f"no_{metadata['family']}_{metadata['layer']}",
                    losses,
                    chunk,
                )
            context["scales"] = []

            original = snapshot_states(modules)
            latest = writer_updates(modules, previous_mem)
            record_state_geometry(geometry_records, modules, latest, chunk)
            variants = {
                "state_zero_matrix": [torch.zeros_like(x) for x in original],
                "state_latest_update_only": latest,
                "state_without_latest_update": [
                    state - update for state, update in zip(original, latest)
                ],
                "state_first_update_only": first_update_states,
                "state_cross_example_shuffled": [
                    state.roll(1, 0) for state in original
                ],
            }
            for name, values in variants.items():
                restore_states(modules, values)
                losses, _ = forward_losses(
                    model, input_ids, labels, previous_mem, use_autocast
                )
                append_loss_windows(loss_records, name, losses, chunk)
            restore_states(modules, original)

        if chunk < len(input_chunks) - 1:
            model.increment_state(new_mem)
            if chunk == 0:
                first_update_states = snapshot_states(modules)
        previous_mem = new_mem
        print(
            f"pointwise chunk {chunk:02d}, selected={selected}, "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    model.empty_state()
    context["scales"] = []
    context["capture_activations"] = False
    return pd.DataFrame(loss_records), pd.DataFrame(geometry_records)


@torch.inference_mode()
def rollout_analysis(
    model,
    context,
    tokens,
    selected_chunks,
    use_autocast,
):
    device = torch.device("cuda")
    tokens = tokens.to(device)
    chunk_length = int(model.chunk_length)
    input_chunks = list(tokens[:, :-1].split(chunk_length, dim=1))
    label_chunks = list(tokens[:, 1:].split(chunk_length, dim=1))
    conditions = {
        "rollout_baseline": [],
        "rollout_no_all_state": [({"kind": "identity"}, 0.0)],
        "rollout_no_causal_state": [({"kind": "causal"}, 0.0)],
        "rollout_no_backbone_state": [
            ({"kind": "family", "family": "backbone"}, 0.0)
        ],
        "rollout_no_output_state": [
            ({"kind": "family", "family": "output"}, 0.0)
        ],
        "rollout_no_memory_layer_state": [
            ({"kind": "family", "family": "memory"}, 0.0)
        ],
        "rollout_causal_state_scale_2": [({"kind": "causal"}, 2.0)],
    }
    records = []
    started = time.time()
    for condition, scales in conditions.items():
        model.init_state(tokens.shape[0], device)
        model.empty_state()
        previous_mem = None
        context["scales"] = scales
        context["capture_activations"] = False
        for chunk, (input_ids, labels) in enumerate(
            zip(input_chunks, label_chunks)
        ):
            context["chunk"] = chunk
            losses, new_mem = forward_losses(
                model, input_ids, labels, previous_mem, use_autocast
            )
            if chunk in selected_chunks:
                append_loss_windows(records, condition, losses, chunk)
            if chunk < len(input_chunks) - 1:
                model.increment_state(new_mem)
            previous_mem = new_mem
        model.empty_state()
        print(
            f"{condition} complete, elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    context["scales"] = []
    return pd.DataFrame(records)


def paired_summary(records, baseline_condition):
    keys = ["example", "chunk", "window_tokens"]
    baseline = records[records.condition == baseline_condition][
        keys + ["loss"]
    ].rename(columns={"loss": "baseline_loss"})
    merged = records.merge(baseline, on=keys)
    rows = []
    for (condition, width), frame in merged.groupby(
        ["condition", "window_tokens"]
    ):
        delta = frame.loss - frame.baseline_loss
        rows.append(
            {
                "condition": condition,
                "window_tokens": width,
                "mean_loss": float(frame.loss.mean()),
                "baseline_loss": float(frame.baseline_loss.mean()),
                "delta_vs_baseline": float(delta.mean()),
                "delta_sem": float(delta.std(ddof=1) / math.sqrt(len(delta))),
                "fraction_improved": float((delta < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_scale_sweep(summary, output):
    mapping = {
        "state_scale_-1": -1.0,
        "state_scale_0": 0.0,
        "state_scale_0.5": 0.5,
        "baseline": 1.0,
        "state_scale_2": 2.0,
        "state_scale_4": 4.0,
    }
    frame = summary[summary.condition.isin(mapping)].copy()
    frame["scale"] = frame.condition.map(mapping)
    fig, ax = plt.subplots(figsize=(8, 5))
    for width, values in frame.groupby("window_tokens"):
        if width not in (1, 32, 128, 1024):
            continue
        values = values.sort_values("scale")
        ax.plot(values.scale, values.mean_loss, marker="o", label=f"first {width}")
    ax.axvline(1, color="black", linewidth=1, alpha=0.4)
    ax.set(
        xlabel="causal state residual multiplier",
        ylabel="mean next-token loss",
        title="Is the matrix-state residual underweighted?",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_layer_effects(summary, output):
    frame = summary[
        summary.condition.str.match(r"no_(backbone|output)_\d+$")
        & summary.window_tokens.isin([1, 32, 1024])
    ].copy()
    parts = frame.condition.str.extract(r"no_(backbone|output)_(\d+)")
    frame["family"] = parts[0]
    frame["layer"] = parts[1].astype(int)
    frame["x"] = frame.layer + (frame.family == "output") * 18
    fig, ax = plt.subplots(figsize=(11, 5))
    for width, values in frame.groupby("window_tokens"):
        ax.plot(
            values.x,
            values.delta_vs_baseline,
            marker="o",
            label=f"first {width}",
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(17.5, color="black", linestyle="--", alpha=0.35)
    ax.set(
        xlabel="causal layer (0–17 backbone, 18–19 output)",
        ylabel="loss increase when state residual is removed",
        title="Direct usefulness of each state mechanism",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data_config = OmegaConf.load(constants.CONFIG_PATH(args.data_config))
    trainer_config = OmegaConf.load(constants.CONFIG_PATH(args.trainer_config))
    output = args.output_dir / (
        f"{args.checkpoint.replace('/', '--')}_step={args.checkpoint_step}"
        f"_n={args.num_examples}"
    )
    output.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens(data_config, args.num_examples)
    model = load_checkpoint(
        args.checkpoint,
        args.checkpoint_step,
        attention_kernel="gpu_flash_attention",
    )
    model.to(device="cuda", dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules = state_modules(model)
    context = {"capture_activations": False, "scales": [], "chunk": -1}
    activation_records = []
    component_records = []
    handles = register_state_hooks(modules, context, activation_records)
    handles.extend(register_component_hooks(model, context, component_records))
    try:
        pointwise, geometry = pointwise_analysis(
            model,
            modules,
            context,
            tokens,
            set(args.selected_chunks),
            bool(trainer_config.use_autocast),
        )
        rollout = rollout_analysis(
            model,
            context,
            tokens,
            set(args.selected_chunks),
            bool(trainer_config.use_autocast),
        )
    finally:
        for handle in handles:
            handle.remove()

    activations = pd.DataFrame(activation_records)
    components = pd.DataFrame(component_records)
    pointwise_summary = paired_summary(pointwise, "baseline")
    rollout_summary = paired_summary(rollout, "rollout_baseline")
    pointwise.to_csv(output / "pointwise_loss_windows.csv", index=False)
    pointwise_summary.to_csv(output / "pointwise_summary.csv", index=False)
    rollout.to_csv(output / "rollout_loss_windows.csv", index=False)
    rollout_summary.to_csv(output / "rollout_summary.csv", index=False)
    activations.to_csv(output / "state_activation_magnitudes.csv", index=False)
    components.to_csv(output / "component_activation_magnitudes.csv", index=False)
    geometry.to_csv(output / "state_geometry.csv", index=False)
    plot_scale_sweep(pointwise_summary, output / "state_scale_sweep.png")
    plot_layer_effects(pointwise_summary, output / "layer_ablation_effects.png")

    selected_widths = [1, 32, 128, 1024]
    machine_summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "num_examples": args.num_examples,
        "selected_chunks": args.selected_chunks,
        "pointwise": pointwise_summary[
            pointwise_summary.window_tokens.isin(selected_widths)
        ].to_dict(orient="records"),
        "rollout": rollout_summary[
            rollout_summary.window_tokens.isin(selected_widths)
        ].to_dict(orient="records"),
        "activation_by_family": (
            activations.groupby(["family", "window_tokens"], as_index=False)
            .agg(
                state_output_rms=("state_output_rms", "mean"),
                out_scale=("out_scale", "mean"),
            )
            .to_dict(orient="records")
        ),
        "geometry_by_family": (
            geometry.groupby("family", as_index=False)
            .agg(
                raw_state_rms=("raw_state_rms", "mean"),
                latest_update_rms=("latest_update_rms", "mean"),
                latest_to_state_norm_ratio=("latest_to_state_norm_ratio", "mean"),
                raw_state_latest_cosine=("raw_state_latest_cosine", "mean"),
                effective_latest_cosine=("effective_latest_cosine", "mean"),
                raw_state_cross_example_cosine=(
                    "raw_state_cross_example_cosine", "mean"
                ),
                latest_update_cross_example_cosine=(
                    "latest_update_cross_example_cosine", "mean"
                ),
                effective_state_cross_example_cosine=(
                    "effective_state_cross_example_cosine", "mean"
                ),
                effective_state_rms=("effective_state_rms", "mean"),
            )
            .to_dict(orient="records")
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(machine_summary, indent=2) + "\n"
    )
    print(json.dumps(machine_summary, indent=2), flush=True)
    print(f"Wrote state reliance analysis to {output}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
