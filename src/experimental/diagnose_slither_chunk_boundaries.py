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


ATTN_RE = re.compile(
    r"^(backbone|output)_layers\.layers\.(\d+)"
    r"\.self_attn\.attention_block$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablate Slither memory/state pathways at chunk boundaries."
    )
    parser.add_argument("--checkpoint", default="aklein4/slither_alpha-350m")
    parser.add_argument("--checkpoint-step", type=int, default=500)
    parser.add_argument("--data-config", default="data/longattn-smollm2.yaml")
    parser.add_argument("--trainer-config", default="trainer/slither-med.yaml")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument(
        "--selected-chunks",
        type=int,
        nargs="+",
        default=[1, 8, 16, 24, 30],
    )
    parser.add_argument("--attention-probe-chunk", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=constants.LOCAL_DATA_PATH / "slither_boundary_diagnosis",
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


def state_output_zero_hook(_module, _inputs, output):
    return torch.zeros_like(output)


def state_ablation_handles(model):
    return [
        module.register_forward_hook(state_output_zero_hook)
        for module in model.modules()
        if isinstance(module, SlitherStateMechanism)
    ]


def run_forward_loss(
    model,
    input_ids,
    labels,
    mem_states,
    use_autocast,
    disable_state=False,
    logits_slice=None,
):
    handles = state_ablation_handles(model) if disable_state else []
    try:
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            logits, new_mem = model(
                input_ids=input_ids,
                mem_states=mem_states,
            )
        if logits_slice is not None:
            logits = logits[:, logits_slice]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        return losses.float(), new_mem.float()
    finally:
        for handle in handles:
            handle.remove()


def shuffle_states(model, shift):
    with torch.no_grad():
        for mechanism in model._mechanisms():
            mechanism.state.copy_(mechanism.state.roll(shifts=shift, dims=0))


def register_attention_diagnostics(model, context, records):
    handles = []
    local_positions = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1023)

    for name, module in model.named_modules():
        match = ATTN_RE.match(name)
        if match is None:
            continue
        family = match.group(1)
        layer = int(match.group(2))

        def pre_hook(
            _module,
            args,
            *,
            family=family,
            layer=layer,
        ):
            if not context["capture_attention"]:
                return
            query, key, value = args[:3]
            batch, num_heads, full_length, head_dim = query.shape
            memory_length = context["memory_length"]
            if key.shape[1] != num_heads:
                repeats = num_heads // key.shape[1]
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)

            for local_position in local_positions:
                if memory_length + local_position >= full_length:
                    continue
                allowed = memory_length + local_position + 1
                q = query[:, :, memory_length + local_position, :].float()
                k = key[:, :, :allowed, :].float()
                scores = torch.einsum("bhd,bhkd->bhk", q, k) / math.sqrt(
                    head_dim
                )
                weights = torch.softmax(scores, dim=-1)
                memory_mass = weights[:, :, :memory_length].sum(-1)
                local_mass = weights[:, :, memory_length:].sum(-1)
                self_weight = weights[:, :, -1]
                memory_key_norm = key[:, :, :memory_length].float().norm(dim=-1).mean(-1)
                local_key_norm = key[:, :, memory_length:allowed].float().norm(dim=-1).mean(-1)
                memory_value_norm = (
                    value[:, :, :memory_length].float().norm(dim=-1).mean(-1)
                )
                local_value_norm = (
                    value[:, :, memory_length:allowed].float().norm(dim=-1).mean(-1)
                )
                arrays = {
                    "memory_attention_mass": memory_mass.cpu().numpy(),
                    "local_attention_mass": local_mass.cpu().numpy(),
                    "self_attention_weight": self_weight.cpu().numpy(),
                    "memory_key_norm": memory_key_norm.cpu().numpy(),
                    "local_key_norm": local_key_norm.cpu().numpy(),
                    "memory_value_norm": memory_value_norm.cpu().numpy(),
                    "local_value_norm": local_value_norm.cpu().numpy(),
                }

                for example in range(batch):
                    for head in range(num_heads):
                        records.append(
                            {
                                "example": example,
                                "family": family,
                                "layer": layer,
                                "head": head,
                                "within_chunk_position": local_position,
                                **{
                                    column: float(values[example, head])
                                    for column, values in arrays.items()
                                },
                            }
                        )

        handles.append(module.register_forward_pre_hook(pre_hook))
    return handles


@torch.inference_mode()
def capture_ablation_losses(
    model,
    tokens,
    selected_chunks,
    attention_probe_chunk,
    use_autocast,
):
    device = torch.device("cuda")
    tokens = tokens.to(device)
    chunk_length = int(model.chunk_length)
    input_chunks = list(tokens[:, :-1].split(chunk_length, dim=1))
    label_chunks = list(tokens[:, 1:].split(chunk_length, dim=1))
    model.init_state(tokens.shape[0], device)
    previous_mem = None
    loss_records = []
    attention_records = []
    context = {"capture_attention": False, "memory_length": chunk_length}
    attention_handles = register_attention_diagnostics(
        model, context, attention_records
    )
    started = time.time()

    try:
        for chunk, (input_ids, labels) in enumerate(
            zip(input_chunks, label_chunks)
        ):
            selected = chunk in selected_chunks
            context["capture_attention"] = (
                selected and chunk == attention_probe_chunk
            )
            baseline_loss, baseline_new_mem = run_forward_loss(
                model,
                input_ids,
                labels,
                previous_mem,
                use_autocast,
            )
            context["capture_attention"] = False

            if selected:
                conditions = {"baseline": baseline_loss}

                no_state_loss, _ = run_forward_loss(
                    model,
                    input_ids,
                    labels,
                    previous_mem,
                    use_autocast,
                    disable_state=True,
                )
                conditions["memory_only_no_matrix_state"] = no_state_loss

                empty_mem = previous_mem[:, :0]
                state_only_loss, _ = run_forward_loss(
                    model,
                    input_ids,
                    labels,
                    empty_mem,
                    use_autocast,
                )
                conditions["matrix_state_only_no_memory_tokens"] = state_only_loss

                neither_loss, _ = run_forward_loss(
                    model,
                    input_ids,
                    labels,
                    None,
                    use_autocast,
                )
                conditions["neither_memory_nor_matrix_state"] = neither_loss

                shuffled_memory_loss, _ = run_forward_loss(
                    model,
                    input_ids,
                    labels,
                    previous_mem.roll(shifts=1, dims=0),
                    use_autocast,
                )
                conditions["cross_example_shuffled_memory"] = shuffled_memory_loss

                shuffle_states(model, shift=1)
                try:
                    shuffled_state_loss, _ = run_forward_loss(
                        model,
                        input_ids,
                        labels,
                        previous_mem,
                        use_autocast,
                    )
                finally:
                    shuffle_states(model, shift=-1)
                conditions["cross_example_shuffled_matrix_state"] = (
                    shuffled_state_loss
                )

                boundary = chunk * chunk_length
                dense_input = tokens[
                    :, boundary - chunk_length : boundary + chunk_length
                ]
                dense_labels = tokens[
                    :,
                    boundary + 1 : boundary + chunk_length + 1,
                ]
                dense_loss, _ = run_forward_loss(
                    model,
                    dense_input,
                    dense_labels,
                    None,
                    use_autocast,
                    logits_slice=slice(chunk_length, None),
                )
                conditions["dense_raw_previous_chunk_oracle"] = dense_loss

                for condition, values in conditions.items():
                    values = values.cpu().numpy()
                    for example in range(values.shape[0]):
                        for position in range(values.shape[1]):
                            loss_records.append(
                                {
                                    "condition": condition,
                                    "example": example,
                                    "chunk": chunk,
                                    "within_chunk_position": position,
                                    "loss": float(values[example, position]),
                                }
                            )

            context["capture_attention"] = False
            if chunk < len(input_chunks) - 1:
                model.increment_state(baseline_new_mem)
            previous_mem = baseline_new_mem
            print(
                f"chunk {chunk:02d}/{len(input_chunks) - 1:02d}, "
                f"selected={selected}, elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        for handle in attention_handles:
            handle.remove()
        model.empty_state()

    return pd.DataFrame(loss_records), pd.DataFrame(attention_records)


def summarize_conditions(records):
    rows = []
    baseline = records[records.condition == "baseline"]
    key_columns = ["example", "chunk", "within_chunk_position"]
    baseline = baseline[key_columns + ["loss"]].rename(
        columns={"loss": "baseline_loss"}
    )
    merged = records.merge(baseline, on=key_columns)

    for condition, frame in merged.groupby("condition"):
        for width in (1, 8, 32, 64, 128, 256, 1024):
            window = frame[frame.within_chunk_position < width]
            per_pair = window.groupby(["example", "chunk"], as_index=False).agg(
                loss=("loss", "mean"),
                baseline_loss=("baseline_loss", "mean"),
            )
            delta = per_pair["loss"] - per_pair["baseline_loss"]
            rows.append(
                {
                    "condition": condition,
                    "window_tokens": width,
                    "mean_loss": float(per_pair["loss"].mean()),
                    "baseline_mean_loss": float(
                        per_pair["baseline_loss"].mean()
                    ),
                    "delta_vs_baseline": float(delta.mean()),
                    "delta_sem": float(
                        delta.std(ddof=1) / np.sqrt(len(delta))
                    ),
                    "fraction_worse_than_baseline": float(np.mean(delta > 0)),
                }
            )
    return pd.DataFrame(rows)


def plot_condition_windows(summary, output):
    frame = summary[summary.window_tokens < 1024]
    fig, ax = plt.subplots(figsize=(11, 6))
    for condition, values in frame.groupby("condition"):
        ax.plot(
            values["window_tokens"],
            values["mean_loss"],
            marker=".",
            label=condition.replace("_", " "),
        )
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="prefix window size (tokens)",
        ylabel="mean next-token loss",
        title="Boundary-prefix loss under pathway ablations",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_attention_mass(attention, output):
    frame = (
        attention.groupby(["family", "layer", "within_chunk_position"], as_index=False)
        .memory_attention_mass.mean()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, family in zip(axes, ("backbone", "output")):
        family_frame = frame[frame.family == family]
        for layer, values in family_frame.groupby("layer"):
            ax.plot(
                values["within_chunk_position"],
                values["memory_attention_mass"],
                marker=".",
                label=f"layer {layer}",
                alpha=0.8,
            )
        ax.set_xscale("symlog", linthresh=1)
        ax.set(
            xlabel="within-chunk query position",
            title=f"{family.capitalize()} layers",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("attention probability on encoded memory")
    fig.suptitle("Joint-softmax allocation between memory and live-token keys")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
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

    records, attention = capture_ablation_losses(
        model,
        tokens,
        set(args.selected_chunks),
        args.attention_probe_chunk,
        bool(trainer_config.use_autocast),
    )
    summary = summarize_conditions(records)
    records.to_csv(output / "ablation_token_losses.csv", index=False)
    summary.to_csv(output / "ablation_window_summary.csv", index=False)
    attention.to_csv(output / "attention_stream_diagnostics.csv", index=False)
    plot_condition_windows(summary, output / "ablation_prefix_losses.png")
    plot_attention_mass(attention, output / "memory_attention_mass.png")

    selected = summary[summary.window_tokens.isin([1, 32, 128, 1024])]
    machine_summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "dataset": data_config.dataset.url,
        "num_examples": args.num_examples,
        "selected_chunks": args.selected_chunks,
        "attention_probe_chunk": args.attention_probe_chunk,
        "ablation_windows": selected.to_dict(orient="records"),
        "attention_mass_by_position": (
            attention.groupby("within_chunk_position")
            .memory_attention_mass.mean()
            .to_dict()
        ),
        "key_value_norms": {
            column: float(attention[column].mean())
            for column in (
                "memory_key_norm",
                "local_key_norm",
                "memory_value_norm",
                "local_value_norm",
            )
        },
    }
    (output / "summary.json").write_text(
        json.dumps(machine_summary, indent=2) + "\n"
    )
    print(json.dumps(machine_summary, indent=2), flush=True)
    print(f"Wrote diagnosis to {output}", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
