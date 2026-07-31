from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from analyze_slither_mesa_v2 import load_tokens, module_metadata
from models import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    config = OmegaConf.load(args.data_config)
    tokens = load_tokens(config, args.num_examples).to("cuda")
    model = load_checkpoint(
        "aklein4/slither_mesa-v2-350m", 250,
        attention_kernel="gpu_flash_attention",
    ).to("cuda", dtype=torch.float32).eval()
    states, _ = module_metadata(model)
    samples = defaultdict(list)
    context = {"chunk": -1}
    handles = []
    for metadata in states:
        mechanism = metadata["module"]
        key = (metadata["family"], metadata["layer"])

        def hook(_module, _inputs, output, *, key=key):
            # One observation per actual token and layer: average over read heads.
            values = (2.0 * torch.sigmoid(output.float())).mean(-1)
            samples[key].append(values.detach().cpu().numpy().reshape(-1))

        handles.append(mechanism.out_gate.register_forward_hook(hook))

    model.init_state(tokens.shape[0], torch.device("cuda"))
    model.empty_state()
    previous_mem = None
    try:
        for chunk, input_ids in enumerate(tokens[:, :-1].split(model.chunk_length, 1)):
            context["chunk"] = chunk
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, new_mem = model(input_ids, mem_states=previous_mem)
            new_mem = new_mem.float()
            if chunk < 31:
                model.increment_state(new_mem)
            previous_mem = new_mem
            print(f"chunk {chunk:02d}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        model.empty_state()

    samples = {key: np.concatenate(parts) for key, parts in samples.items()}
    bins = np.linspace(0, 2, 101)
    ordered = sorted(samples, key=lambda x: ({"backbone": 0, "output": 1, "memory": 2}[x[0]], x[1]))
    all_values = np.stack([samples[key] for key in ordered]).mean(0)
    rows = []
    global_counts, edges = np.histogram(all_values, bins=bins)
    for left, right, count in zip(edges[:-1], edges[1:], global_counts):
        rows.append({
            "family": "global", "layer": -1,
            "bin_left": left, "bin_right": right, "count": count,
        })
    for (family, layer), values in samples.items():
        counts, edges = np.histogram(values, bins=bins)
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            rows.append({
                "family": family, "layer": layer,
                "bin_left": left, "bin_right": right, "count": count,
            })
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "token_mean_read_out_histogram.csv", index=False)

    colors = {"backbone": "tab:blue", "output": "tab:orange", "memory": "tab:green"}
    fig, axes = plt.subplots(5, 5, figsize=(17, 15), sharex=True)
    axes = axes.flat
    axes[0].hist(all_values, bins=bins, density=True, color="0.25")
    axes[0].axvline(all_values.mean(), color="crimson", lw=1.5)
    axes[0].set_title(f"Global (mean={all_values.mean():.3f})")
    for ax, key in zip(axes[1:], ordered):
        values = samples[key]
        ax.hist(values, bins=bins, density=True, color=colors[key[0]])
        ax.axvline(values.mean(), color="crimson", lw=1)
        ax.set_title(f"{key[0]} {key[1]}  mean={values.mean():.3f}")
    for ax in axes:
        ax.set_xlim(0, 2)
        ax.grid(alpha=0.15)
    fig.supxlabel("per-token mean read-out gate over heads (2 sigmoid)")
    fig.supylabel("density")
    fig.tight_layout()
    fig.savefig(args.output / "token_mean_read_out_histograms.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame([{
        "family": "global", "layer": -1, "count": len(all_values),
        "mean": all_values.mean(), "std": all_values.std(),
        "q01": np.quantile(all_values, .01), "q50": np.quantile(all_values, .5),
        "q99": np.quantile(all_values, .99),
    }] + [
        {"family": family, "layer": layer, "count": len(values),
         "mean": values.mean(), "std": values.std(),
         "q01": np.quantile(values, .01), "q50": np.quantile(values, .5),
         "q99": np.quantile(values, .99)}
        for (family, layer), values in samples.items()
    ]).sort_values(["family", "layer"])
    summary.to_csv(args.output / "token_mean_read_out_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
