from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from collators.single_sequence import SingleSequenceCollator
from data.datasets import get_dataset
from models import load_checkpoint
from models.slither import _pcg_solve_forward


DEFAULT_STEPS = (0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure PCG convergence on real Slither checkpoint systems."
    )
    parser.add_argument(
        "--checkpoint", default="aklein4/slither_mesa-350m"
    )
    parser.add_argument("--checkpoint-step", type=int, default=250)
    parser.add_argument(
        "--dataset", default="aklein4/longattn-SmolLM2"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-count", type=int, default=64)
    parser.add_argument(
        "--query-chunk",
        type=int,
        default=1,
        help="Zero-based chunk whose PCG systems should be measured.",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=DEFAULT_STEPS
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pcg_convergence_step250.png"),
    )
    return parser.parse_args()


def layer_names(model) -> list[str]:
    names = []
    names.extend(
        f"backbone {index}"
        for index, _ in enumerate(model.backbone_layers._iter_layers())
    )
    names.extend(
        f"output {index}"
        for index, _ in enumerate(model.output_layers._iter_layers())
    )
    names.extend(
        f"memory {index}"
        for index, _ in enumerate(model.memory_layers._iter_layers())
    )
    return names


@torch.no_grad()
def capture_systems(model, input_ids, device, query_chunk=1):
    if query_chunk < 1:
        raise ValueError("query_chunk must be at least 1")
    model.init_state(input_ids.shape[0], device)

    memory = None
    for chunk_index in range(query_chunk):
        start = chunk_index * model.chunk_length
        stop = start + model.chunk_length
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, memory = model(
                input_ids[:, start:stop],
                mem_states=memory,
                skip_logits=True,
            )
        memory = memory.float()
        model.increment_state(memory)

    captured = []
    hooks = []

    def capture(module, inputs, kwargs):
        hidden_states = kwargs.get("hidden_states", inputs[0] if inputs else None)
        query = module.activation(module.q_proj(hidden_states))
        gate = (
            torch.softmax(module.in_gate(hidden_states), dim=-1)
            * module.num_state_in_heads
        )
        query = module.in_norm(query, scales=gate)

        batch_size, sequence_length, _ = query.shape
        rhs = query.view(
            batch_size,
            sequence_length,
            module.num_state_in_heads,
            module.in_head_dim,
        ).permute(0, 2, 3, 1)

        count = module.k_count.clamp_min(1).to(module.k_corr.dtype)
        correlation = module.k_corr / count[:, None, None, None]
        identity = torch.eye(
            module.in_head_dim,
            device=device,
            dtype=correlation.dtype,
        )
        matrix = (
            correlation
            + module.get_lambda()[None, :, :, None]
            * identity[None, None]
        )
        captured.append(
            {
                "matrix": matrix.detach().float().cpu(),
                "rhs": rhs.detach().float().cpu(),
                "regularizer": module.get_lambda().detach().float().cpu(),
            }
        )

    for mechanism in model._mechanisms():
        hooks.append(
            mechanism.register_forward_pre_hook(capture, with_kwargs=True)
        )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        start = query_chunk * model.chunk_length
        stop = start + model.chunk_length
        model(
            input_ids[:, start:stop],
            mem_states=memory,
            skip_logits=True,
        )

    for hook in hooks:
        hook.remove()

    return captured


@torch.no_grad()
def measure_system(system, steps, query_count, device):
    matrix = system["matrix"].to(device)
    rhs = system["rhs"].to(device)
    query_indices = torch.linspace(
        0,
        rhs.shape[-1] - 1,
        query_count,
        device=device,
    ).round().long()
    rhs = rhs.index_select(-1, query_indices)

    # FP64 is used only to establish a reference for these small 128x128
    # systems. PCG below receives the original FP32 matrix and right-hand side.
    exact = torch.linalg.solve(matrix.double(), rhs.double())
    exact_norm = exact.norm(dim=(-2, -1))
    rhs_norm = rhs.double().norm(dim=(-2, -1))

    solution_errors = []
    residual_errors = []
    for iteration_count in steps:
        if iteration_count == 0:
            solution = torch.zeros_like(rhs)
        else:
            solution = _pcg_solve_forward(
                matrix, rhs, iteration_count, eps=1e-5
            )
        solution = solution.double()
        residual = matrix.double() @ solution - rhs.double()
        solution_errors.append(
            ((solution - exact).norm(dim=(-2, -1)) / exact_norm).cpu()
        )
        residual_errors.append(
            (residual.norm(dim=(-2, -1)) / rhs_norm).cpu()
        )

    eigenvalues = torch.linalg.eigvalsh(matrix.double()).cpu()
    condition = eigenvalues[..., -1] / eigenvalues[..., 0]
    return {
        "solution_error": torch.stack(solution_errors, dim=-1),
        "residual_error": torch.stack(residual_errors, dim=-1),
        "condition": condition,
    }


def summarize(
    layer_names_,
    measurements,
    systems,
    steps,
    query_chunk,
    batch_size,
    query_count,
):
    solution = torch.stack(
        [measurement["solution_error"] for measurement in measurements]
    )
    residual = torch.stack(
        [measurement["residual_error"] for measurement in measurements]
    )
    condition = torch.stack(
        [measurement["condition"] for measurement in measurements]
    )

    rows = []
    for step_index, step in enumerate(steps):
        values = solution[..., step_index].flatten()
        residual_values = residual[..., step_index].flatten()
        rows.append(
            {
                "steps": step,
                "solution_error_median": values.median().item(),
                "solution_error_p90": values.quantile(0.9).item(),
                "solution_error_max": values.max().item(),
                "residual_error_median": residual_values.median().item(),
                "residual_error_p90": residual_values.quantile(0.9).item(),
                "residual_error_max": residual_values.max().item(),
            }
        )

    regularizers = torch.cat(
        [system["regularizer"].flatten() for system in systems]
    )
    return {
        "measurement": {
            "query_chunk_zero_based": query_chunk,
            "preceding_chunks_in_state": query_chunk,
            "batch_size": batch_size,
            "queries_per_sequence": query_count,
        },
        "layers": layer_names_,
        "steps": list(steps),
        "aggregate": rows,
        "condition_number": {
            "median": condition.median().item(),
            "p90": condition.flatten().quantile(0.9).item(),
            "max": condition.max().item(),
        },
        "regularizer": {
            "min": regularizers.min().item(),
            "median": regularizers.median().item(),
            "max": regularizers.max().item(),
        },
        "solution_by_layer": solution.reshape(
            solution.shape[0], -1, solution.shape[-1]
        ).median(dim=1).values.tolist(),
        "residual_by_layer": residual.reshape(
            residual.shape[0], -1, residual.shape[-1]
        ).median(dim=1).values.tolist(),
    }


def plot(summary, output):
    steps = np.asarray(summary["steps"])
    solution = np.asarray(summary["solution_by_layer"])
    residual = np.asarray(summary["residual_by_layer"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    panels = (
        (axes[0], solution, "Relative solution error", r"$||x_k-x^*||/||x^*||$"),
        (axes[1], residual, "Relative residual", r"$||Ax_k-b||/||b||$"),
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, solution.shape[0]))

    for axis, values, title, ylabel in panels:
        for layer_index, layer_values in enumerate(values):
            axis.plot(
                steps,
                layer_values,
                color=colors[layer_index],
                alpha=0.35,
                linewidth=1,
            )
        axis.plot(
            steps,
            np.median(values, axis=0),
            color="black",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="median layer",
        )
        axis.axvline(10, color="#d62728", linestyle="--", label="training: 10")
        axis.axvline(30, color="#1f77b4", linestyle=":", label="MesaNet: 30")
        axis.set_xscale("symlog", linthresh=1)
        axis.set_yscale("log")
        axis.set_xlabel("PCG iterations")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

    metadata = summary["measurement"]
    fig.suptitle(
        "Slither PCG convergence on step-250 checkpoint systems\n"
        f"(chunk index {metadata['query_chunk_zero_based']}, "
        f"{metadata['batch_size']} real training sequences, "
        f"{metadata['queries_per_sequence']} sampled queries, 24 layers)"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU.")
    device = torch.device("cuda")

    model = load_checkpoint(
        args.checkpoint,
        args.checkpoint_step,
        attention_kernel="gpu_flash_attention",
    ).to(device)
    model.eval()

    dataset = get_dataset(
        args.dataset,
        {"split": "train", "streaming": True},
    )
    rows = []
    iterator = iter(dataset)
    for _ in range(args.batch_size):
        rows.append(next(iterator))
    del iterator, dataset
    gc.collect()
    collator = SingleSequenceCollator(
        sequence_length=(args.query_chunk + 1) * model.chunk_length,
        pad_token_id=model.config.pad_token_id,
        vocab_size=model.config.vocab_size,
    )
    input_ids = collator(rows)["input_ids"].to(device)

    systems = capture_systems(
        model,
        input_ids,
        device,
        query_chunk=args.query_chunk,
    )
    names = layer_names(model)
    measurements = [
        measure_system(
            system,
            args.steps,
            args.query_count,
            device,
        )
        for system in systems
    ]
    summary = summarize(
        names,
        measurements,
        systems,
        args.steps,
        query_chunk=args.query_chunk,
        batch_size=args.batch_size,
        query_count=args.query_count,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(summary, args.output)
    json_output = args.output.with_suffix(".json")
    json_output.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["aggregate"], indent=2))
    print(json.dumps(summary["condition_number"], indent=2))
    print(json.dumps(summary["regularizer"], indent=2))
    print(f"Saved {args.output} and {json_output}")

    # `datasets` streaming leaves a PyArrow worker alive in this environment,
    # which can abort CPython during interpreter teardown after all outputs
    # have been written. Exit directly after flushing on the successful path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
