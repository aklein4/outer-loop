from __future__ import annotations

import argparse
import gc
from pathlib import Path
import re

from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn.functional as F

from data.datasets import get_dataset
from models import load_checkpoint
from utils.import_utils import import_collator


STATE_RE = re.compile(r"^(backbone|output|memory)_layers\.layers\.(\d+)\.state_mechanism$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--num-examples", type=int, default=4)
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


def cosine(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


@torch.inference_mode()
def main():
    args = parse_args()
    config = OmegaConf.load(args.data_config)
    tokens = load_tokens(config, args.num_examples).to("cuda")
    model = load_checkpoint(
        "aklein4/slither_mesa-v2-350m",
        250,
        attention_kernel="gpu_flash_attention",
    ).to("cuda", dtype=torch.float32).eval()
    selected = []
    wanted = {
        ("backbone", 0), ("backbone", 8), ("backbone", 15),
        ("output", 0), ("output", 2), ("output", 3), ("memory", 3),
    }
    for name, module in model.named_modules():
        match = STATE_RE.match(name)
        if match and (match.group(1), int(match.group(2))) in wanted:
            selected.append((match.group(1), int(match.group(2)), module))

    model.init_state(tokens.shape[0], torch.device("cuda"))
    model.empty_state()
    previous_mem = None
    previous_updates = {}
    rows = []
    for chunk, input_ids in enumerate(tokens[:, :-1].split(model.chunk_length, 1)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, new_mem = model(input_ids, mem_states=previous_mem)
        new_mem = new_mem.float()
        for family, layer, module in selected:
            update, _, _ = module.writer(new_mem)
            accumulated = module.state
            acc_cos = cosine(update, accumulated) if chunk else torch.full(
                (update.shape[0],), float("nan"), device=update.device
            )
            key = (family, layer)
            prev_cos = (
                cosine(update, previous_updates[key])
                if key in previous_updates
                else torch.full((update.shape[0],), float("nan"), device=update.device)
            )
            cross_cos = F.cosine_similarity(
                update[0].flatten()[None], update[1:].flatten(1), dim=1
            ).mean() if update.shape[0] > 1 else torch.tensor(float("nan"), device=update.device)
            for example in range(update.shape[0]):
                rows.append(
                    {
                        "chunk": chunk,
                        "family": family,
                        "layer": layer,
                        "example": example,
                        "update_accumulated_cosine": float(acc_cos[example]),
                        "successive_update_cosine": float(prev_cos[example]),
                        "cross_example_update_cosine": float(cross_cos),
                        "update_to_accumulated_norm": float(
                            update[example].norm() / accumulated[example].norm().clamp_min(1e-12)
                        ) if chunk else float("nan"),
                    }
                )
            previous_updates[key] = update.clone()
        model.increment_state(new_mem)
        previous_mem = new_mem
        print(f"chunk {chunk:02d}", flush=True)

    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "update_geometry.csv", index=False)
    mature = frame[frame.chunk >= 8]
    summary = mature.groupby(["family", "layer"], as_index=False).agg(
        accumulated_cosine=("update_accumulated_cosine", "mean"),
        successive_cosine=("successive_update_cosine", "mean"),
        cross_example_cosine=("cross_example_update_cosine", "mean"),
        relative_update_norm=("update_to_accumulated_norm", "mean"),
    )
    summary.to_csv(args.output / "update_geometry_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
