from __future__ import annotations

import argparse
import gc
from pathlib import Path
import re

from omegaconf import OmegaConf
import pandas as pd
import torch

from data.datasets import get_dataset
from models import load_checkpoint
from utils.import_utils import import_collator


STATE_RE = re.compile(r"^(backbone|output|memory)_layers\.layers\.(\d+)\.state_mechanism$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="aklein4/slither_mesa-v2-350m")
    parser.add_argument("--step", type=int, default=250)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument("--chunks", type=int, nargs="+", default=[0, 1, 8, 16, 30])
    parser.add_argument("--output", type=Path, required=True)
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


@torch.inference_mode()
def main():
    args = parse_args()
    config = OmegaConf.load(args.data_config)
    tokens = load_tokens(config, args.num_examples).to("cuda")
    model = load_checkpoint(
        args.checkpoint, args.step, attention_kernel="gpu_flash_attention"
    ).to("cuda", dtype=torch.float32).eval()
    mechanisms = []
    for name, module in model.named_modules():
        match = STATE_RE.match(name)
        if match:
            mechanisms.append((match.group(1), int(match.group(2)), module))

    model.init_state(tokens.shape[0], torch.device("cuda"))
    model.empty_state()
    previous_mem = None
    rows = []
    inputs = list(tokens[:, :-1].split(model.chunk_length, 1))
    for chunk, input_ids in enumerate(inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, new_mem = model(input_ids, mem_states=previous_mem)
        new_mem = new_mem.float()
        model.increment_state(new_mem)
        if chunk in args.chunks:
            for family, layer, module in mechanisms:
                count = module.k_count.clamp_min(1).to(module.k_corr.dtype)
                corr = module.k_corr / count[:, None, None, None]
                ridge = module.get_lambda()[None].expand(corr.shape[0], -1, -1)
                matrix = corr + torch.diag_embed(ridge)
                corr_eigs = torch.linalg.eigvalsh(corr.float())
                matrix_eigs = torch.linalg.eigvalsh(matrix.float())
                for example in range(corr.shape[0]):
                    for head in range(corr.shape[1]):
                        raw_max = float(corr_eigs[example, head, -1])
                        effective_min = float(matrix_eigs[example, head, 0])
                        effective_max = float(matrix_eigs[example, head, -1])
                        rows.append(
                            {
                                "chunk_after_write": chunk,
                                "family": family,
                                "layer": layer,
                                "example": example,
                                "head": head,
                                "count": float(count[example]),
                                "corr_min_eigenvalue": float(corr_eigs[example, head, 0]),
                                "corr_max_eigenvalue": raw_max,
                                "ridge_mean": float(ridge[example, head].mean()),
                                "ridge_min": float(ridge[example, head].min()),
                                "ridge_max": float(ridge[example, head].max()),
                                "effective_min_eigenvalue": effective_min,
                                "effective_max_eigenvalue": effective_max,
                                "effective_condition": effective_max / effective_min,
                                "ridge_to_corr_max": float(ridge[example, head].mean()) / max(raw_max, 1e-12),
                            }
                        )
        previous_mem = new_mem
        print(f"chunk {chunk:02d}", flush=True)

    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "conditioning_by_head.csv", index=False)
    summary = frame.groupby(
        ["chunk_after_write", "family"], as_index=False
    ).agg(
        condition_median=("effective_condition", "median"),
        condition_p95=("effective_condition", lambda x: x.quantile(0.95)),
        condition_max=("effective_condition", "max"),
        ridge_mean=("ridge_mean", "mean"),
        ridge_to_corr_max_median=("ridge_to_corr_max", "median"),
        effective_min_median=("effective_min_eigenvalue", "median"),
        effective_max_median=("effective_max_eigenvalue", "median"),
    )
    summary.to_csv(args.output / "conditioning_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
