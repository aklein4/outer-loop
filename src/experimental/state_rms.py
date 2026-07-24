import argparse
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import torch
from tqdm import trange

from evaluate_icl import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    DEFAULT_TOKENIZER,
    adaptation_loss,
    autocast,
    encode,
    load_model,
    load_tokenizer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-step", type=int, default=250)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("state_rms.png"))
    args = parser.parse_args()

    device = torch.device(args.device)
    subsets = [args.subset] if args.subset else datasets.get_dataset_config_names(args.dataset)
    rows = []
    for subset in subsets:
        dataset = datasets.load_dataset(args.dataset, subset, split="train")
        rows += [row for row in dataset if len(row["train_data"]) >= 1024]
        if len(rows) >= args.batch_size:
            break
    rows = rows[:args.batch_size]
    if len(rows) < args.batch_size:
        raise ValueError(f"Only found {len(rows)} rows with 1024 examples")

    model = load_model(args.checkpoint, args.checkpoint_step, device)
    tokenizer = load_tokenizer(args.tokenizer)
    model.init_state(args.batch_size, device)
    model.empty_state()

    rms = []
    for step in trange(1024, desc="Updating fast weights"):
        input_ids, assistant_mask, attention_mask = encode(
            tokenizer, [row["train_data"][step] for row in rows], args.max_length, device
        )
        with torch.enable_grad(), autocast(device, args.dtype):
            logits = model(input_ids, logits_to_keep=slice(0, -1))[0]
            loss = adaptation_loss(input_ids, assistant_mask, attention_mask, logits, 0.0)
        loss.backward()
        model.update_state()
        rms.append([
            layer.mlp.fast.state.float().square().mean((-2, -1)).sqrt().mean().item()
            for layer in model.model.layers
        ])

    rms = torch.tensor(rms)
    rms /= rms.amax(dim=0, keepdim=True).clamp_min(1e-12)
    colors = plt.cm.viridis(torch.linspace(0, 1, rms.shape[1]))
    for layer, color in enumerate(colors):
        plt.plot(range(1, 1025), rms[:, layer], color=color)
    plt.xlabel("Update step")
    plt.ylabel("RMS / maximum RMS")
    plt.title(f"Fast-weight state RMS (checkpoint step {args.checkpoint_step})")
    colorbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, rms.shape[1])),
        ax=plt.gca(),
    )
    colorbar.set_label("Layer")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
