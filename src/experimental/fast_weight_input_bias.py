import argparse
import csv
import json
import os
import sys
from itertools import islice
from pathlib import Path

# Avoid paying torch.compile startup cost for the decorated update helpers during
# a statistics-only pass.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import datasets
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from models.ittt.configuration_ittt import ItttConfig
from models.ittt.modelling_ittt import FastWeight, ItttModel
from utils.import_utils import import_collator


DEFAULT_REPO = "aklein4/iTTT-Cluster_horizons-corr-init-v2"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ittt_smollm2_360m_horizons.yaml"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "local_data" / "fast_weight_input_bias_step_1000.png"


class LayerMoments:
    def __init__(self, n_layers: int, width: int, device: torch.device):
        self.count = [torch.zeros((), dtype=torch.float64, device=device) for _ in range(n_layers)]
        self.sum = [torch.zeros(width, dtype=torch.float64, device=device) for _ in range(n_layers)]
        self.sum_sq = [torch.zeros(width, dtype=torch.float64, device=device) for _ in range(n_layers)]

    @torch.no_grad()
    def update(self, layer_idx: int, x: torch.Tensor, mask: torch.Tensor | None) -> None:
        x = x.detach().float()
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.float32)
            x = x * mask[..., None]
            count = mask.sum()
        else:
            count = torch.tensor(x.shape[0] * x.shape[1], dtype=torch.float32, device=x.device)

        self.count[layer_idx].add_(count.to(torch.float64))
        self.sum[layer_idx].add_(x.sum(dim=(0, 1), dtype=torch.float32).to(torch.float64))
        self.sum_sq[layer_idx].add_(x.square().sum(dim=(0, 1), dtype=torch.float32).to(torch.float64))

    def finalize(self, eps: float) -> list[dict[str, float]]:
        rows = []
        for layer_idx, (count, total, total_sq) in enumerate(zip(self.count, self.sum, self.sum_sq)):
            count_cpu = count.cpu()
            total_cpu = total.cpu()
            total_sq_cpu = total_sq.cpu()

            mean = total_cpu / count_cpu.clamp_min(1.0)
            variance = total_sq_cpu / count_cpu.clamp_min(1.0) - mean.square()
            std = variance.clamp_min(0.0).sqrt()
            ratio = mean.abs() / std.clamp_min(eps)

            rows.append(
                {
                    "layer": layer_idx,
                    "tokens": int(count_cpu.item()),
                    "mean_abs_mean_over_std": float(ratio.mean().item()),
                    "max_abs_mean_over_std": float(ratio.max().item()),
                    "median_abs_mean_over_std": float(ratio.median().item()),
                }
            )
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate FastWeight.forward input bias by layer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-repo", default=DEFAULT_REPO)
    parser.add_argument("--checkpoint-step", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-clusters", type=int, default=0)
    parser.add_argument("--max-horizons", type=int, default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--autocast-dtype", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--normal-exit", action="store_true", help="Do not hard-exit after streaming dataset cleanup.")
    return parser.parse_args()


def load_model(repo: str, step: int, attn_implementation: str | None, device: torch.device) -> ItttModel:
    subfolder = f"step_{step:012d}"
    config = ItttConfig.from_pretrained(repo, subfolder=subfolder)
    if attn_implementation and attn_implementation.lower() != "none":
        config._attn_implementation = attn_implementation

    print(f"Loading model config from {repo}/{subfolder}")
    model = ItttModel(config)

    state_path = hf_hub_download(repo, filename=f"{subfolder}/model.safetensors")
    print(f"Loading checkpoint weights from {state_path}")
    state = load_file(state_path, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
        for key in missing[:10]:
            print(f"  missing: {key}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for key in unexpected[:10]:
            print(f"  unexpected: {key}")

    model.to(device=device, dtype=torch.float32)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def cluster_loss(input_ids: torch.Tensor, assistant_mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = assistant_mask[:, 1:].float().contiguous()
    losses = F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    return ((losses * shift_mask).sum(1) / shift_mask.sum(1).clamp(min=1)).mean()


def register_hooks(model: ItttModel, moments: LayerMoments, current_mask: dict[str, torch.Tensor | None]):
    handles = []
    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, FastWeight):
            parts = name.split(".")
            if "layers" in parts:
                layer_idx = int(parts[parts.index("layers") + 1])
            else:
                layer_idx = len(layer_names)
            layer_names.append(name)

            def hook(_module, inputs, idx=layer_idx):
                moments.update(idx, inputs[0], current_mask["value"])

            handles.append(module.register_forward_pre_hook(hook))

    if not layer_names:
        raise RuntimeError("No FastWeight modules found.")
    return handles, layer_names


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


def write_metadata(path: Path, metadata: dict) -> None:
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {metadata_path}")


def plot_rows(path: Path, rows: list[dict[str, float]], metadata: dict) -> None:
    layers = [row["layer"] for row in rows]
    mean_values = [row["mean_abs_mean_over_std"] for row in rows]
    max_values = [row["max_abs_mean_over_std"] for row in rows]
    median_values = [row["median_abs_mean_over_std"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(layers, mean_values, marker="o", linewidth=2, label="mean_i")
    ax.plot(layers, median_values, marker="o", linewidth=2, label="median_i")
    ax.plot(layers, max_values, marker="o", linewidth=2, label="max_i")
    ax.set_xlabel("Model layer")
    ax.set_ylabel("abs(mean(x_i)) / std(x_i)")
    ax.set_title(
        "FastWeight input activation bias "
        f"(step {metadata['checkpoint_step']}, {metadata['num_clusters']} clusters)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    autocast_dtype = args.autocast_dtype or cfg.trainer.autocast_dtype
    autocast_dtype = getattr(torch, autocast_dtype)

    model = load_model(args.checkpoint_repo, args.checkpoint_step, args.attn_implementation, device)
    n_layers = len(model.llama.model.layers)
    width = model.config.fast_weight_size
    moments = LayerMoments(n_layers=n_layers, width=width, device=device)

    current_mask = {"value": None}
    handles, layer_names = register_hooks(model, moments, current_mask)
    print(f"Registered hooks for {len(layer_names)} FastWeight modules")

    collator = import_collator(cfg.collator.type)(**cfg.collator.kwargs)
    dataset = datasets.load_dataset(cfg.dataset.name, split=cfg.dataset.kwargs.split, streaming=True)
    iterator = iter(dataset)
    if args.skip_clusters:
        iterator = islice(iterator, args.skip_clusters, None)

    processed_clusters = 0
    processed_horizon_batches = 0
    model.init_state(args.batch_size, device)

    try:
        pbar = tqdm(total=args.num_clusters, desc="Collecting clusters")
        while processed_clusters < args.num_clusters:
            raw_batch = list(islice(iterator, args.batch_size))
            if not raw_batch:
                break
            if len(raw_batch) != args.batch_size:
                print(f"Stopping on partial batch of size {len(raw_batch)}")
                break

            batch = collator(raw_batch)
            input_ids = batch["input_ids"]
            assistant_mask = batch["assistant_mask"]
            attention_mask = batch["attention_mask"]

            _, horizon_length, _ = input_ids.shape
            if args.max_horizons is not None:
                horizon_length = min(horizon_length, args.max_horizons)

            model.empty_state()
            for horizon_idx in range(horizon_length):
                example_ids = input_ids[:, horizon_idx, :]
                example_assistant_mask = assistant_mask[:, horizon_idx, :]
                current_mask["value"] = attention_mask[:, horizon_idx, :]

                with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    logits = model(example_ids, logits_to_keep=slice(0, -1)).logits
                    loss = cluster_loss(example_ids, example_assistant_mask, logits)

                loss.backward()
                model.update_state()
                model.zero_grad(set_to_none=True)
                processed_horizon_batches += 1

            processed_clusters += len(raw_batch)
            pbar.update(len(raw_batch))
        pbar.close()
    finally:
        current_mask["value"] = None
        for handle in handles:
            handle.remove()
        model.empty_state()

    rows = moments.finalize(args.eps)
    metadata = {
        "checkpoint_repo": args.checkpoint_repo,
        "checkpoint_step": args.checkpoint_step,
        "config": str(args.config),
        "dataset": cfg.dataset.name,
        "split": cfg.dataset.kwargs.split,
        "num_clusters": processed_clusters,
        "batch_size": args.batch_size,
        "skip_clusters": args.skip_clusters,
        "processed_horizon_batches": processed_horizon_batches,
        "processed_cluster_horizons": processed_horizon_batches * args.batch_size,
        "max_horizons": args.max_horizons,
        "mask": "attention_mask",
        "fast_weight_modules": layer_names,
        "autocast_dtype": str(autocast_dtype).replace("torch.", ""),
    }

    plot_rows(args.output, rows, metadata)
    write_csv(args.output, rows)
    write_metadata(args.output, metadata)
    print(json.dumps(rows, indent=2))
    sys.stdout.flush()

    if not args.normal_exit:
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
