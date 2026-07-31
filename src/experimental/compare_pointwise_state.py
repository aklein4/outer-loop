from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn.functional as F

from data.datasets import get_dataset
from models import load_checkpoint
from utils.import_utils import import_collator


WINDOWS = (1, 8, 32, 128, 1024)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-solve", action="store_true")
    parser.add_argument("--gate-ablations", action="store_true")
    parser.add_argument("--parameter-ablations", action="store_true")
    parser.add_argument("--plot-with", type=Path)
    parser.add_argument("--plot-output", type=Path)
    return parser.parse_args()


def load_tokens(config, count):
    dataset = get_dataset(config.dataset.url, config.dataset.kwargs)
    iterator = iter(dataset)
    rows = [next(iterator) for _ in range(count)]
    collator = import_collator(config.collator.type)(**config.collator.kwargs)
    tokens = collator(rows)["input_ids"]
    del iterator, dataset, rows
    gc.collect()
    return tokens


def state_mechanisms(model):
    mechanisms = []
    for name, module in model.named_modules():
        if hasattr(module, "increment_state") and hasattr(module, "state_size"):
            if name.endswith("state_mechanism"):
                mechanisms.append((name, module))
    if not mechanisms:
        raise RuntimeError("No state mechanisms found")
    return mechanisms


def shuffle_state(mechanisms, shift):
    with torch.no_grad():
        for _, module in mechanisms:
            for field in ("state", "k_corr", "k_count"):
                value = getattr(module, field, None)
                if isinstance(value, torch.Tensor):
                    value.copy_(value.roll(shift, 0))


def set_solve(mechanisms, enabled):
    changed = 0
    for _, module in mechanisms:
        if hasattr(module, "mse_solve"):
            module.mse_solve = enabled
            changed += 1
    return changed


def register_gate_ablation_hooks(model, context):
    handles = []
    for name, module in model.named_modules():
        if name.startswith("output_layers.layers.") and name.endswith(
            ".state_mechanism.out_gate"
        ):
            def state_hook(_module, _inputs, output):
                if context["condition"] == "output_read_gate_neutral":
                    return torch.zeros_like(output)
                if context["condition"] == "output_read_gate_fixed_open":
                    return torch.full_like(output, 20.0)
                return None

            handles.append(module.register_forward_hook(state_hook))

        if name == "output_layers.layers.0.self_attn.gate_proj":
            def attention_hook(_module, _inputs, output):
                if context["condition"] != "saturated_attention_head_neutral":
                    return None
                result = output.clone()
                result[..., 2] = 0
                return result

            handles.append(module.register_forward_hook(attention_hook))
    return handles


def add_records(records, condition, chunk, losses):
    for requested_width in WINDOWS:
        actual_width = min(requested_width, losses.shape[1])
        values = losses[:, :actual_width].mean(1).detach().float().cpu()
        for example, value in enumerate(values.tolist()):
            records.append(
                {
                    "condition": condition,
                    "example": example,
                    "chunk": chunk,
                    "window_tokens": requested_width,
                    "actual_window_tokens": actual_width,
                    "loss": value,
                }
            )


@torch.inference_mode()
def run(
    model,
    tokens,
    mechanisms,
    disable_solve,
    gate_ablations,
    parameter_ablations,
):
    device = torch.device("cuda")
    tokens = tokens.to(device)
    inputs = list(tokens[:, :-1].split(model.chunk_length, 1))
    labels = list(tokens[:, 1:].split(model.chunk_length, 1))
    model.init_state(tokens.shape[0], device)
    model.empty_state()
    previous_mem = None
    records = []
    gate_context = {"condition": "baseline"}
    gate_handles = register_gate_ablation_hooks(model, gate_context)
    parameter_copies = {
        name: {
            "odot": module.odot.detach().clone(),
            "log_lambda": module.log_lambda.detach().clone(),
        }
        for name, module in mechanisms
        if hasattr(module, "odot") and hasattr(module, "log_lambda")
    }
    solve_modules = sum(hasattr(module, "mse_solve") for _, module in mechanisms)
    if disable_solve and solve_modules == 0:
        raise RuntimeError("--disable-solve requested but model has no Mesa solve")

    try:
        for chunk, (input_ids, targets) in enumerate(zip(inputs, labels)):
            def forward(condition="baseline"):
                gate_context["condition"] = condition
                changed_parameters = condition in ("odot_zero", "ridge_at_init")
                if changed_parameters:
                    with torch.no_grad():
                        for name, module in mechanisms:
                            if name not in parameter_copies:
                                continue
                            if condition == "odot_zero":
                                module.odot.zero_()
                            elif condition == "ridge_at_init":
                                module.log_lambda.zero_()
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, new_mem = model(input_ids, mem_states=previous_mem)
                finally:
                    if changed_parameters:
                        with torch.no_grad():
                            for name, module in mechanisms:
                                if name not in parameter_copies:
                                    continue
                                module.odot.copy_(parameter_copies[name]["odot"])
                                module.log_lambda.copy_(parameter_copies[name]["log_lambda"])
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).view_as(targets)
                return losses.float(), new_mem.float()

            baseline_losses, new_mem = forward()
            add_records(records, "baseline", chunk, baseline_losses)

            if chunk == 0:
                add_records(records, "shuffled_state", chunk, baseline_losses)
                if disable_solve:
                    add_records(records, "solve_disabled", chunk, baseline_losses)
            else:
                shuffle_state(mechanisms, 1)
                shuffled_losses, _ = forward()
                shuffle_state(mechanisms, -1)
                add_records(records, "shuffled_state", chunk, shuffled_losses)

                if disable_solve:
                    changed = set_solve(mechanisms, False)
                    try:
                        solve_losses, _ = forward()
                    finally:
                        set_solve(mechanisms, True)
                    if changed != solve_modules:
                        raise RuntimeError("Solve-module count changed during run")
                    add_records(records, "solve_disabled", chunk, solve_losses)

            if gate_ablations:
                for condition in (
                    "output_read_gate_neutral",
                    "output_read_gate_fixed_open",
                    "saturated_attention_head_neutral",
                ):
                    losses, _ = forward(condition)
                    add_records(records, condition, chunk, losses)

            if parameter_ablations:
                for condition in ("odot_zero", "ridge_at_init"):
                    losses, _ = forward(condition)
                    add_records(records, condition, chunk, losses)

            if chunk < len(inputs) - 1:
                model.increment_state(new_mem)
            previous_mem = new_mem
            print(f"chunk {chunk:02d}", flush=True)

    finally:
        gate_context["condition"] = "baseline"
        for handle in gate_handles:
            handle.remove()

    model.empty_state()
    return pd.DataFrame(records), solve_modules


def summarize(records):
    keys = ["example", "chunk", "window_tokens"]
    baseline = records[records.condition == "baseline"][keys + ["loss"]].rename(
        columns={"loss": "baseline_loss"}
    )
    paired = records.merge(baseline, on=keys)
    paired["delta"] = paired.loss - paired.baseline_loss
    rows = []
    for (condition, chunk, width), frame in paired.groupby(
        ["condition", "chunk", "window_tokens"]
    ):
        rows.append(
            {
                "condition": condition,
                "chunk": chunk,
                "window_tokens": width,
                "mean_loss": frame.loss.mean(),
                "baseline_loss": frame.baseline_loss.mean(),
                "delta": frame.delta.mean(),
                "delta_sem": frame.delta.std(ddof=1) / math.sqrt(len(frame)),
                "fraction_worse": (frame.delta > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(current, previous, output):
    current = current[current.condition == "shuffled_state"].copy()
    previous = previous[previous.condition == "shuffled_state"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, width in zip(axes.flat, (1, 8, 32, 1024)):
        for frame, label, color in (
            (previous, "alpha step 500", "tab:blue"),
            (current, "mesa-v2 step 250", "tab:orange"),
        ):
            values = frame[frame.window_tokens == width]
            ax.plot(values.chunk, values.delta, marker="o", markersize=3, label=label, color=color)
            ax.fill_between(
                values.chunk,
                values.delta - values.delta_sem,
                values.delta + values.delta_sem,
                color=color,
                alpha=0.15,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"First {width} token{'s' if width != 1 else ''}")
        ax.set_ylabel("loss delta from state shuffle")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("chunk")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_solve_ablation(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, width in zip(axes.flat, (1, 8, 32, 1024)):
        for condition, label, color in (
            ("solve_disabled", "Mesa solve disabled", "tab:red"),
            ("shuffled_state", "State shuffled", "tab:orange"),
        ):
            values = summary[
                (summary.condition == condition)
                & (summary.window_tokens == width)
            ]
            ax.plot(
                values.chunk,
                values.delta,
                marker="o",
                markersize=3,
                label=label,
                color=color,
            )
            ax.fill_between(
                values.chunk,
                values.delta - values.delta_sem,
                values.delta + values.delta_sem,
                color=color,
                alpha=0.15,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"First {width} token{'s' if width != 1 else ''}")
        ax.set_ylabel("loss delta versus baseline")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("chunk")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.time()
    config = OmegaConf.load(args.data_config)
    tokens = load_tokens(config, args.num_examples)
    model = load_checkpoint(
        args.checkpoint,
        args.checkpoint_step,
        attention_kernel="gpu_flash_attention",
    ).to("cuda", dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mechanisms = state_mechanisms(model)
    records, solve_modules = run(
        model,
        tokens,
        mechanisms,
        args.disable_solve,
        args.gate_ablations,
        args.parameter_ablations,
    )
    summary = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output / "records.csv", index=False)
    summary.to_csv(args.output / "by_chunk.csv", index=False)
    metadata = {
        "checkpoint": args.checkpoint,
        "step": args.checkpoint_step,
        "num_examples": args.num_examples,
        "state_mechanisms": len(mechanisms),
        "solve_modules": solve_modules,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.disable_solve:
        plot_solve_ablation(summary, args.output / "solve_disabled_by_chunk.png")
    if args.plot_with is not None:
        if args.plot_output is None:
            raise ValueError("--plot-output is required with --plot-with")
        plot_comparison(summary, pd.read_csv(args.plot_with), args.plot_output)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
