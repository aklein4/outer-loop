from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import pandas as pd
import torch

from analyze_slither_mesa_v2 import load_tokens
from models import load_checkpoint


MODULE_RE = re.compile(
    r"^(backbone|output|memory)_layers\.layers\.(\d+)\.(self_attn|state_mechanism|mlp)$"
)
BRANCH = {"self_attn": "attention", "state_mechanism": "state", "mlp": "mlp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", required=True)
    p.add_argument("--num-examples", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    config = OmegaConf.load(args.data_config)
    tokens = load_tokens(config, args.num_examples).to("cuda")
    model = load_checkpoint(
        "aklein4/slither_mesa-v2-350m", 250,
        attention_kernel="gpu_flash_attention",
    ).to("cuda", dtype=torch.float32).eval()
    records = []
    context = {"chunk": -1}
    handles = []
    for name, module in model.named_modules():
        match = MODULE_RE.match(name)
        if not match:
            continue
        family, layer, module_name = match.group(1), int(match.group(2)), match.group(3)
        branch = BRANCH[module_name]

        def hook(_module, _inputs, output, *, family=family, layer=layer, branch=branch):
            # Exclude chunk zero so all three branches use exactly the same tokens.
            if context["chunk"] == 0:
                return
            x = output.float()
            flat = x.reshape(-1, x.shape[-1])
            token_rms = flat.square().mean(-1).sqrt()
            records.append({
                "chunk": context["chunk"], "family": family,
                "layer": layer, "branch": branch,
                "count": flat.numel(),
                "mean": float(flat.mean()),
                "rms": float(flat.square().mean().sqrt()),
                "variance": float(flat.var(unbiased=False)),
                "token_rms_mean": float(token_rms.mean()),
                "token_rms_std": float(token_rms.std(unbiased=False)),
                "token_rms_q01": float(torch.quantile(token_rms, .01)),
                "token_rms_q50": float(torch.quantile(token_rms, .50)),
                "token_rms_q99": float(torch.quantile(token_rms, .99)),
            })

        handles.append(module.register_forward_hook(hook))

    inputs = list(tokens[:, :-1].split(model.chunk_length, 1))
    model.init_state(tokens.shape[0], torch.device("cuda"))
    model.empty_state()
    previous_mem = None
    try:
        for chunk, input_ids in enumerate(inputs):
            context["chunk"] = chunk
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, new_mem = model(input_ids, mem_states=previous_mem)
            new_mem = new_mem.float()
            if chunk < len(inputs) - 1:
                model.increment_state(new_mem)
            previous_mem = new_mem
            print(f"chunk {chunk:02d}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        model.empty_state()

    frame = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "residual_write_by_chunk.csv", index=False)
    summary = frame.groupby(["family", "layer", "branch"], as_index=False).agg(
        rms=("rms", "mean"), variance=("variance", "mean"),
        token_rms_mean=("token_rms_mean", "mean"),
        token_rms_std=("token_rms_std", "mean"),
        token_rms_q01=("token_rms_q01", "mean"),
        token_rms_q50=("token_rms_q50", "mean"),
        token_rms_q99=("token_rms_q99", "mean"),
        mean=("mean", "mean"),
    )
    summary.to_csv(args.output / "residual_write_summary.csv", index=False)

    colors = {"attention": "tab:blue", "state": "tab:orange", "mlp": "tab:green"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for row, family in enumerate(("backbone", "output", "memory")):
        part = summary[summary.family == family]
        for branch in ("attention", "state", "mlp"):
            b = part[part.branch == branch]
            axes[row, 0].plot(b.layer, b.rms, marker="o", label=branch, color=colors[branch])
            axes[row, 1].plot(b.layer, b.variance, marker="o", label=branch, color=colors[branch])
        axes[row, 0].set(title=f"{family}: residual-write RMS", xlabel="layer", ylabel="RMS")
        axes[row, 1].set(title=f"{family}: residual-write variance", xlabel="layer", ylabel="variance")
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "residual_write_rms_variance.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
