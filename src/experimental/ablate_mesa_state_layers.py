from __future__ import annotations

import argparse
import math
from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn.functional as F

from experimental.analyze_slither_mesa_v2 import load_tokens, module_metadata
from models import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="aklein4/slither_mesa-v2-350m")
    parser.add_argument("--step", type=int, default=250)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 8, 16, 24, 30])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    config = OmegaConf.load(args.data_config)
    tokens = load_tokens(config, args.num_examples).to("cuda")
    model = load_checkpoint(
        args.checkpoint, args.step, attention_kernel="gpu_flash_attention"
    ).to("cuda", dtype=torch.float32).eval()
    states, _ = module_metadata(model)
    causal = [s for s in states if s["family"] != "memory"]
    context = {"target": None}
    handles = []
    for metadata in causal:
        key = f'{metadata["family"]}.{metadata["layer"]}'

        def hook(_module, _inputs, output, *, key=key):
            if context["target"] == key:
                return torch.zeros_like(output)
            return None

        handles.append(metadata["module"].register_forward_hook(hook))

    inputs = list(tokens[:, :-1].split(model.chunk_length, 1))
    labels = list(tokens[:, 1:].split(model.chunk_length, 1))
    model.init_state(tokens.shape[0], torch.device("cuda"))
    model.empty_state()
    previous_mem = None
    records = []

    def forward(input_ids, targets):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, new_mem = model(input_ids, mem_states=previous_mem)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        return losses.float(), new_mem.float()

    try:
        for chunk, (input_ids, targets) in enumerate(zip(inputs, labels)):
            context["target"] = None
            baseline, new_mem = forward(input_ids, targets)
            if chunk in args.chunks:
                for metadata in causal:
                    key = f'{metadata["family"]}.{metadata["layer"]}'
                    context["target"] = key
                    losses, _ = forward(input_ids, targets)
                    for width in (1, 32, 1024):
                        actual = min(width, losses.shape[1])
                        delta = losses[:, :actual].mean(1) - baseline[:, :actual].mean(1)
                        for example, value in enumerate(delta.cpu().tolist()):
                            records.append(
                                {
                                    "family": metadata["family"],
                                    "layer": metadata["layer"],
                                    "example": example,
                                    "chunk": chunk,
                                    "window_tokens": width,
                                    "delta": value,
                                }
                            )
            context["target"] = None
            if chunk < len(inputs) - 1:
                model.increment_state(new_mem)
            previous_mem = new_mem
            print(f"chunk {chunk:02d}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        model.empty_state()

    frame = pd.DataFrame(records)
    summary = frame.groupby(
        ["family", "layer", "window_tokens"], as_index=False
    ).agg(
        delta=("delta", "mean"),
        delta_sem=("delta", lambda x: x.std(ddof=1) / math.sqrt(len(x))),
        fraction_worse=("delta", lambda x: (x > 0).mean()),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "state_layer_ablation_records.csv", index=False)
    summary.to_csv(args.output / "state_layer_ablation_summary.csv", index=False)
    print(summary[summary.window_tokens == 1024].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
