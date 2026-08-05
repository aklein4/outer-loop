"""Measure each ICL training example immediately before and after adaptation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_icl import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_TOKENIZER,
    adaptation_lr_scale,
    adaptation_loss,
    autocast,
    encode,
    load_all_rows,
    load_model,
    load_tokenizer,
    output_loss,
)
from models.forte import ForteMode, ForteModel  # noqa: E402
from utils import constants  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="aklein4/Horizon-TPU_forte-v3-1b")
    parser.add_argument("--checkpoint-step", type=int, default=300)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-adaptation-steps", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--lr-scale-decay", action="store_true")
    parser.add_argument("--lr-scale-start", type=float, default=1.0)
    parser.add_argument("--lr-scale-end", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_training_rows(args: argparse.Namespace) -> list[dict]:
    subsets = args.subsets or datasets.get_dataset_config_names(args.dataset)
    # Reuse the evaluator's trajectory selection, but require no held-out examples.
    loader_args = SimpleNamespace(
        dataset=args.dataset,
        num_examples=[args.num_adaptation_steps],
        num_eval=0,
        max_rows=args.max_rows,
    )
    rows = load_all_rows(loader_args, subsets)
    if not rows:
        raise RuntimeError("No eligible training trajectories were found")
    return rows


def make_step_fns(model, args, device):
    def train_step(input_ids, assistant_mask, attention_mask, lr_scale):
        with torch.no_grad():
            hidden_states = model.forward_backbone(input_ids, mode=ForteMode.INFERENCE)
            embeddings = model.forward_embeddings(hidden_states, attention_mask)
        with autocast(device, args.dtype):
            logits = model(
                input_ids,
                embeddings=embeddings,
                embedding_mask=attention_mask,
                mode=ForteMode.TRAIN_FIRST,
                logits_to_keep=slice(0, -1),
            )
            pre_losses = output_loss(input_ids, assistant_mask, logits)
            loss = adaptation_loss(
                input_ids, assistant_mask, attention_mask, logits, args.aux_weight
            )
        loss.backward()
        model.update_state(
            embeddings,
            attention_mask,
            mode=ForteMode.TRAIN_FIRST,
            lr_scale=lr_scale,
        )
        return pre_losses

    def post_step(input_ids, assistant_mask):
        with autocast(device, args.dtype):
            logits = model(input_ids, logits_to_keep=slice(0, -1))
            return output_loss(input_ids, assistant_mask, logits)

    if args.compile:
        train_step = torch.compile(train_step, fullgraph=False)
        post_step = torch.compile(post_step, fullgraph=False)
    return train_step, post_step


def run_batch(model, train_step, post_step, tokenizer, rows, args, device):
    if not isinstance(model, ForteModel):
        raise TypeError(f"Expected ForteModel, got {type(model).__name__}")
    pre = np.empty((len(rows), args.num_adaptation_steps), dtype=np.float32)
    post = np.empty_like(pre)
    model.init_state(len(rows), device)
    model.empty_state()
    try:
        for step in tqdm(range(args.num_adaptation_steps), desc="adapting", leave=False):
            input_ids, assistant_mask, attention_mask = encode(
                tokenizer,
                [row["train_data"][step] for row in rows],
                args.max_length,
                device,
            )
            lr_scale = adaptation_lr_scale(args, step, args.num_adaptation_steps)
            # A tensor scale avoids specializing the compiled graph on a Python value.
            lr_scale = torch.as_tensor(
                [lr_scale], device=device, dtype=torch.float32
            ).reshape(1, 1, 1)
            with torch.enable_grad():
                pre_losses = train_step(
                    input_ids, assistant_mask, attention_mask, lr_scale
                )
            pre[:, step] = pre_losses.detach().float().cpu().numpy()

            # This is an inference-only second forward over the same example: it
            # observes the new state but performs no backward/update operation.
            with torch.no_grad():
                post_losses = post_step(input_ids, assistant_mask)
            post[:, step] = post_losses.float().cpu().numpy()

            # Embedding parameters participate in backward but are not optimized.
            model.embed_tokens.weight.grad = None
    finally:
        model.empty_state()
        model.embed_tokens.weight.grad = None
    return pre, post


def save_results(output_dir: Path, rows, pre: np.ndarray, post: np.ndarray, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "losses.npz", pre_update_loss=pre, post_update_loss=post)

    with (output_dir / "losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory", "subset", "adaptation_step", "pre_update_loss", "post_update_loss"])
        for trajectory, row in enumerate(rows):
            for step in range(args.num_adaptation_steps):
                writer.writerow([trajectory, row["subset"], step + 1, pre[trajectory, step], post[trajectory, step]])

    metadata = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": args.checkpoint_step,
        "dataset": args.dataset,
        "num_trajectories": len(rows),
        "num_adaptation_steps": args.num_adaptation_steps,
        "batch_size": args.batch_size,
        "compile": args.compile,
        "dtype": args.dtype,
        "device": str(args.device),
        "trajectories": [{"trajectory": i, "subset": row["subset"]} for i, row in enumerate(rows)],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    steps = np.arange(1, args.num_adaptation_steps + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(steps, pre.mean(axis=0), label="Pre-update loss", linewidth=1.6)
    ax.plot(steps, post.mean(axis=0), label="Post-update loss", linewidth=1.6)
    ax.set_xlabel("Adaptation step")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title(f"Training-example loss before and after adaptation (step {args.checkpoint_step})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "average_pre_post_losses.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = (
            constants.LOCAL_DATA_PATH
            / f"forte_v3_step{args.checkpoint_step}_pre_post_losses"
        )
    device = torch.device(args.device)
    rows = load_training_rows(args)
    print(f"Loaded {len(rows)} training trajectories", flush=True)
    model = load_model(args.checkpoint, args.checkpoint_step, device)
    tokenizer = load_tokenizer(args.tokenizer)
    train_step, post_step = make_step_fns(model, args, device)

    all_pre, all_post = [], []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="trajectory batches"):
        pre, post = run_batch(
            model,
            train_step,
            post_step,
            tokenizer,
            rows[start : start + args.batch_size],
            args,
            device,
        )
        all_pre.append(pre)
        all_post.append(post)
    pre = np.concatenate(all_pre, axis=0)
    post = np.concatenate(all_post, axis=0)
    save_results(args.output_dir, rows, pre, post, args)
    print(f"Wrote results and plot to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
