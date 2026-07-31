from __future__ import annotations

import argparse
import gc
from pathlib import Path
import re

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


STATE_RE = re.compile(r"^(backbone|output|memory)_layers\.layers\.(\d+)\.state_mechanism$")
WANTED = {
    ("backbone", 0), ("backbone", 8), ("backbone", 15),
    ("output", 0), ("output", 2), ("output", 3), ("memory", 3),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--spectrum-chunks", type=int, nargs="+", default=[1, 8, 16, 24, 30])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lambda-multipliers", type=float, nargs="+", default=None,
        help="Pointwise counterfactual multipliers for the learned lambda.",
    )
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


def cosine(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


def right_transform(matrix, transform, heads, head_dim):
    shaped = matrix.view(matrix.shape[0], matrix.shape[1], heads, head_dim)
    return torch.einsum("bohi,bhij->bohj", shaped, transform).flatten(2)


def inverse_factors(matrix):
    # SVD is used explicitly to match U S^{-1/2} V^H from the question.
    u, singular, vh = torch.linalg.svd(matrix.float())
    inv_sqrt = torch.einsum(
        "bhij,bhj,bhjk->bhik", u, singular.rsqrt(), vh
    )
    inverse = torch.einsum(
        "bhij,bhj,bhjk->bhik", u, singular.reciprocal(), vh
    )
    return singular, inv_sqrt, inverse


def effective_rank(values):
    probabilities = values / values.sum(-1, keepdim=True).clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    return entropy.exp()


def plot_geometry(summary, output):
    frame = summary.copy()
    frame["global_layer"] = frame.apply(
        lambda row: row.layer + {"backbone": 0, "output": 16, "memory": 20}[row.family],
        axis=1,
    )
    frame = frame.sort_values("global_layer")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, suffix, title in (
        (axes[0], "update_accumulated_cosine", "Update versus accumulated state"),
        (axes[1], "successive_cosine", "Successive chunk updates"),
    ):
        columns = (
            (f"raw_{suffix}", "raw", "tab:blue"),
            (
                "whitened_update_accumulated_cosine"
                if suffix == "update_accumulated_cosine"
                else "common_cumulative_whitened_successive_cosine",
                "inverse square-root whitened",
                "tab:orange",
            ),
            (
                "solved_update_accumulated_cosine"
                if suffix == "update_accumulated_cosine"
                else "common_cumulative_solved_successive_cosine",
                "full inverse solved",
                "tab:green",
            ),
        )
        for column, label, color in columns:
            ax.plot(frame.global_layer, frame[column], marker="o", label=label, color=color)
        ax.set(title=title, xlabel="global layer index", ylabel="cosine similarity")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_spectra(spectra, output):
    columns = (
        ("corr_singular_value", "raw C"),
        ("system_singular_value", "C + lambda I"),
        ("whitened_corr_singular_value", "W^T C W"),
        ("whitened_system_singular_value", "W^T (C + lambda I) W"),
    )
    chunks = [1, 8, 16, 30]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, chunk in zip(axes.flat, chunks):
        frame = spectra[spectra.chunk == chunk]
        medians = frame.groupby("index")[[column for column, _ in columns]].median()
        for column, label in columns:
            ax.plot(medians.index, medians[column], label=label)
        ax.set_yscale("log")
        ax.set(title=f"After {chunk} prior chunk{'s' if chunk != 1 else ''}", xlabel="singular-value index", ylabel="singular value")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_lambda_sweep(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for column, label in (
        ("whitened_update_accumulated_cosine", "update vs state"),
        ("whitened_successive_cosine", "successive updates"),
    ):
        axes[0].plot(summary.lambda_multiplier, summary[column], marker="o", label=label)
    for column, label in (
        ("solved_update_accumulated_cosine", "update vs state"),
        ("solved_successive_cosine", "successive updates"),
    ):
        axes[1].plot(summary.lambda_multiplier, summary[column], marker="o", label=label)
    for ax, title in zip(axes, ("Inverse square-root metric", "Full inverse-solve metric")):
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set(xlabel="learned lambda multiplier", ylabel="mean cosine", title=title)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


@torch.inference_mode()
def main():
    args = parse_args()
    config = OmegaConf.load(args.data_config)
    tokens = load_tokens(config, args.num_examples).to("cuda")
    model = load_checkpoint(
        "aklein4/slither_mesa-v2-350m", 250,
        attention_kernel="gpu_flash_attention",
    ).to("cuda", dtype=torch.float32).eval()
    selected = []
    for name, module in model.named_modules():
        match = STATE_RE.match(name)
        if match and (match.group(1), int(match.group(2))) in WANTED:
            selected.append((match.group(1), int(match.group(2)), module))

    model.init_state(tokens.shape[0], torch.device("cuda"))
    model.empty_state()
    previous_mem = None
    previous_updates = {}
    geometry_rows = []
    spectrum_rows = []
    spectrum_summary_rows = []
    lambda_sweep_rows = []

    for chunk, input_ids in enumerate(tokens[:, :-1].split(model.chunk_length, 1)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, new_mem = model(input_ids, mem_states=previous_mem)
        new_mem = new_mem.float()

        for family, layer, module in selected:
            update, update_corr, update_count = module.writer(new_mem)
            key = (family, layer)
            if chunk > 0:
                count = module.k_count.clamp_min(1).to(module.k_corr.dtype)
                corr = module.k_corr / count[:, None, None, None]
                ridge = torch.diag_embed(module.get_lambda())[None]
                system = corr + ridge
                system_sv, inv_sqrt, inverse = inverse_factors(system)

                whitened_update = right_transform(
                    update, inv_sqrt, module.num_state_in_heads, module.in_head_dim
                )
                whitened_accumulated = right_transform(
                    module.state, inv_sqrt, module.num_state_in_heads, module.in_head_dim
                )
                solved_update = right_transform(
                    update, inverse, module.num_state_in_heads, module.in_head_dim
                )
                solved_accumulated = right_transform(
                    module.state, inverse, module.num_state_in_heads, module.in_head_dim
                )

                chunk_corr = update_corr / update_count.to(update_corr.dtype)[:, None, None, None]
                _, chunk_inv_sqrt, _ = inverse_factors(chunk_corr + ridge)
                intrinsically_whitened_update = right_transform(
                    update, chunk_inv_sqrt,
                    module.num_state_in_heads, module.in_head_dim,
                )

                raw_acc_cos = cosine(update, module.state)
                white_acc_cos = cosine(whitened_update, whitened_accumulated)
                solved_acc_cos = cosine(solved_update, solved_accumulated)
                if key in previous_updates:
                    previous = previous_updates[key]
                    raw_prev_cos = cosine(update, previous["raw"])
                    white_prev_cos = cosine(
                        whitened_update,
                        right_transform(
                            previous["raw"], inv_sqrt,
                            module.num_state_in_heads, module.in_head_dim,
                        ),
                    )
                    intrinsic_prev_cos = cosine(
                        intrinsically_whitened_update,
                        previous["intrinsic_whitened"],
                    )
                    solved_prev_cos = cosine(
                        solved_update,
                        right_transform(
                            previous["raw"], inverse,
                            module.num_state_in_heads, module.in_head_dim,
                        ),
                    )
                else:
                    nan = torch.full((update.shape[0],), float("nan"), device=update.device)
                    raw_prev_cos = white_prev_cos = intrinsic_prev_cos = solved_prev_cos = nan

                if args.lambda_multipliers is not None:
                    learned_lambda = module.get_lambda()
                    for multiplier in args.lambda_multipliers:
                        intervened_system = corr + torch.diag_embed(
                            learned_lambda * multiplier
                        )[None]
                        singular, counterfactual_inv_sqrt, counterfactual_inverse = inverse_factors(
                            intervened_system
                        )
                        cf_white_update = right_transform(
                            update, counterfactual_inv_sqrt,
                            module.num_state_in_heads, module.in_head_dim,
                        )
                        cf_white_state = right_transform(
                            module.state, counterfactual_inv_sqrt,
                            module.num_state_in_heads, module.in_head_dim,
                        )
                        cf_solved_update = right_transform(
                            update, counterfactual_inverse,
                            module.num_state_in_heads, module.in_head_dim,
                        )
                        cf_solved_state = right_transform(
                            module.state, counterfactual_inverse,
                            module.num_state_in_heads, module.in_head_dim,
                        )
                        cf_white_acc = cosine(cf_white_update, cf_white_state)
                        cf_solved_acc = cosine(cf_solved_update, cf_solved_state)
                        if key in previous_updates:
                            previous_raw = previous_updates[key]["raw"]
                            cf_white_prev = cosine(
                                cf_white_update,
                                right_transform(previous_raw, counterfactual_inv_sqrt,
                                                module.num_state_in_heads, module.in_head_dim),
                            )
                            cf_solved_prev = cosine(
                                cf_solved_update,
                                right_transform(previous_raw, counterfactual_inverse,
                                                module.num_state_in_heads, module.in_head_dim),
                            )
                        else:
                            cf_white_prev = cf_solved_prev = nan
                        for example in range(update.shape[0]):
                            lambda_sweep_rows.append({
                                "chunk": chunk, "family": family, "layer": layer,
                                "example": example, "lambda_multiplier": multiplier,
                                "lambda_mean": float(learned_lambda.mean()),
                                "system_min_singular": float(singular[example].min()),
                                "system_condition": float(
                                    singular[example, :, 0].max()
                                    / singular[example, :, -1].min().clamp_min(1e-30)
                                ),
                                "whitened_update_accumulated_cosine": float(cf_white_acc[example]),
                                "solved_update_accumulated_cosine": float(cf_solved_acc[example]),
                                "whitened_successive_cosine": float(cf_white_prev[example]),
                                "solved_successive_cosine": float(cf_solved_prev[example]),
                            })

                for example in range(update.shape[0]):
                    geometry_rows.append(
                        {
                            "chunk": chunk,
                            "family": family,
                            "layer": layer,
                            "example": example,
                            "raw_update_accumulated_cosine": float(raw_acc_cos[example]),
                            "whitened_update_accumulated_cosine": float(white_acc_cos[example]),
                            "solved_update_accumulated_cosine": float(solved_acc_cos[example]),
                            "raw_successive_cosine": float(raw_prev_cos[example]),
                            "common_cumulative_whitened_successive_cosine": float(white_prev_cos[example]),
                            "per_chunk_whitened_successive_cosine": float(intrinsic_prev_cos[example]),
                            "common_cumulative_solved_successive_cosine": float(solved_prev_cos[example]),
                        }
                    )

                if chunk in args.spectrum_chunks:
                    whitened_corr = inv_sqrt.mT @ corr @ inv_sqrt
                    whitened_system = inv_sqrt.mT @ system @ inv_sqrt
                    corr_sv = torch.linalg.svdvals(corr.float())
                    white_corr_sv = torch.linalg.svdvals(whitened_corr.float())
                    white_system_sv = torch.linalg.svdvals(whitened_system.float())
                    for example in range(corr.shape[0]):
                        for head in range(corr.shape[1]):
                            for index in range(corr.shape[-1]):
                                spectrum_rows.append(
                                    {
                                        "chunk": chunk,
                                        "family": family,
                                        "layer": layer,
                                        "example": example,
                                        "head": head,
                                        "index": index,
                                        "corr_singular_value": float(corr_sv[example, head, index]),
                                        "system_singular_value": float(system_sv[example, head, index]),
                                        "whitened_corr_singular_value": float(white_corr_sv[example, head, index]),
                                        "whitened_system_singular_value": float(white_system_sv[example, head, index]),
                                    }
                                )
                    for name, values in (
                        ("corr", corr_sv),
                        ("system", system_sv),
                        ("whitened_corr", white_corr_sv),
                        ("whitened_system", white_system_sv),
                    ):
                        rank = effective_rank(values)
                        for example in range(values.shape[0]):
                            for head in range(values.shape[1]):
                                spectrum_summary_rows.append(
                                    {
                                        "chunk": chunk,
                                        "family": family,
                                        "layer": layer,
                                        "example": example,
                                        "head": head,
                                        "matrix": name,
                                        "max": float(values[example, head, 0]),
                                        "median": float(values[example, head].median()),
                                        "min": float(values[example, head, -1]),
                                        "condition": float(
                                            values[example, head, 0]
                                            / values[example, head, -1].clamp_min(1e-30)
                                        ),
                                        "effective_rank": float(rank[example, head]),
                                    }
                                )

                previous_updates[key] = {
                    "raw": update.clone(),
                    "intrinsic_whitened": intrinsically_whitened_update.clone(),
                }
            else:
                chunk_corr = update_corr / update_count.to(update_corr.dtype)[:, None, None, None]
                ridge = torch.diag_embed(module.get_lambda())[None]
                _, chunk_inv_sqrt, _ = inverse_factors(chunk_corr + ridge)
                previous_updates[key] = {
                    "raw": update.clone(),
                    "intrinsic_whitened": right_transform(
                        update, chunk_inv_sqrt,
                        module.num_state_in_heads, module.in_head_dim,
                    ).clone(),
                }

        model.increment_state(new_mem)
        previous_mem = new_mem
        print(f"chunk {chunk:02d}", flush=True)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    geometry = pd.DataFrame(geometry_rows)
    spectra = pd.DataFrame(spectrum_rows)
    spectrum_summary = pd.DataFrame(spectrum_summary_rows)
    geometry.to_csv(output / "whitened_update_geometry.csv", index=False)
    spectra.to_csv(output / "key_correlation_spectra.csv", index=False)
    spectrum_summary.to_csv(output / "key_correlation_spectrum_summary.csv", index=False)

    mature = geometry[geometry.chunk >= 8]
    columns = [c for c in geometry.columns if c.endswith("cosine")]
    geometry_summary = mature.groupby(["family", "layer"], as_index=False)[columns].mean()
    geometry_summary.to_csv(output / "whitened_update_geometry_summary.csv", index=False)
    plot_geometry(geometry_summary, output / "whitened_update_cosines.png")
    plot_spectra(spectra, output / "key_correlation_spectra.png")
    if lambda_sweep_rows:
        lambda_sweep = pd.DataFrame(lambda_sweep_rows)
        lambda_sweep.to_csv(output / "lambda_intervention_geometry.csv", index=False)
        lambda_summary = lambda_sweep[lambda_sweep.chunk >= 8].groupby(
            "lambda_multiplier", as_index=False
        ).agg(
            lambda_mean=("lambda_mean", "mean"),
            system_min_singular=("system_min_singular", "median"),
            system_condition=("system_condition", "median"),
            whitened_update_accumulated_cosine=("whitened_update_accumulated_cosine", "mean"),
            solved_update_accumulated_cosine=("solved_update_accumulated_cosine", "mean"),
            whitened_successive_cosine=("whitened_successive_cosine", "mean"),
            solved_successive_cosine=("solved_successive_cosine", "mean"),
        )
        lambda_summary.to_csv(output / "lambda_intervention_summary.csv", index=False)
        plot_lambda_sweep(lambda_summary, output / "lambda_intervention_cosines.png")
        print("\nLambda intervention summary (chunks 8-31):", flush=True)
        print(lambda_summary.to_string(index=False), flush=True)
    print(geometry_summary.to_string(index=False), flush=True)
    print(
        spectrum_summary.groupby(["chunk", "matrix"])[
            ["max", "median", "min", "condition", "effective_rank"]
        ].median().to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
