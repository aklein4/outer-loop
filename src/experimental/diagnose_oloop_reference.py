from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import datasets
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from collators.horizon import HorizonCollator
from models.oloop import FastWeight, FastWeightMLP, OLoopModel
from utils.torch_modules import enable_gradient_checkpointing


def clean_state(path: Path):
    return {
        key.replace("_orig_mod.", ""): value
        for key, value in torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ).items()
    }


def load_model(
    config_path: Path,
    state_path: Path,
    base_state_path: Path | None,
    batch_size: int,
):
    config = OmegaConf.load(config_path)
    config.attention_kernel = "gpu_flash_attention"
    torch.manual_seed(42)
    model = OLoopModel(config)
    state = clean_state(base_state_path or state_path)
    model.load_state_dict(state, strict=base_state_path is None)
    del state
    gc.collect()
    model.to("cuda")
    model.lm_head.to(torch.bfloat16)
    model.train()
    enable_gradient_checkpointing(model)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.model.embed_tokens.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)
    model.init_state(batch_size, torch.device("cuda"))
    model.zero_grad(set_to_none=False)
    return model


def load_batch(batch_size: int, cluster_length: int):
    dataset = datasets.load_dataset(
        "aklein4/horizons-10B",
        split="train",
        streaming=True,
    )
    iterator = iter(dataset)
    rows = [next(iterator) for _ in range(batch_size)]
    collator = HorizonCollator(
        "meta-llama/Llama-3.2-1B-Instruct",
        max_length=1024,
        cluster_length=cluster_length,
    )
    return {
        key: value.to("cuda")
        for key, value in collator(rows).items()
    }


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def loss(model, input_ids, assistant_mask):
    with autocast():
        logits = model(
            input_ids,
            logits_to_keep=slice(0, -1),
        )[0].float()
    labels = input_ids[:, 1:]
    mask = assistant_mask[:, 1:].float()
    token_losses = F.cross_entropy(
        logits.flatten(0, 1),
        labels.flatten(),
        reduction="none",
    ).reshape_as(labels)
    return (
        (token_losses * mask).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1.0)
    ).mean()


def classify_parameter(name: str):
    if "fast" in name:
        return "fast"
    if "self_attn" in name:
        return "slow_attention"
    if ".mlp." in name:
        return "slow_mlp"
    return "slow_norm"


@torch.no_grad()
def gradient_summary(model):
    groups = {}
    layers = {}
    total_square = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        square = parameter.grad.detach().float().square().sum().item()
        total_square += square
        family = classify_parameter(name)
        groups[family] = groups.get(family, 0.0) + square
        if ".layers." in name:
            layer = name.split(".layers.", 1)[1].split(".", 1)[0]
            layers[layer] = layers.get(layer, 0.0) + square
    return {
        "total_norm": math.sqrt(total_square),
        "families": {
            key: math.sqrt(value)
            for key, value in groups.items()
        },
        "layers": {
            key: math.sqrt(value)
            for key, value in layers.items()
        },
    }


def describe(values):
    values = sorted(values)
    return {
        "min": values[0],
        "median": values[len(values) // 2],
        "max": values[-1],
    }


@torch.no_grad()
def update_with_stats(model, episode):
    modules = [
        module
        for module in model.modules()
        if isinstance(module, FastWeight)
    ]
    before = [module.state.clone() for module in modules]
    rows = []
    for layer, module in enumerate(modules):
        rows.append(
            {
                "episode": episode,
                "layer": layer,
                "raw_grad_rms": (
                    module.momentum.grad.float().square().mean().sqrt().item()
                ),
                "momentum_rms": (
                    module.momentum.float().square().mean().sqrt().item()
                ),
                "state_rms_before": (
                    module.state.float().square().mean().sqrt().item()
                ),
                "effective_state_rms_before": (
                    module.get_s().float().square().mean().sqrt().item()
                ),
                "lr_rms": (
                    module.get_lr().float().square().mean().sqrt().item()
                ),
                "lr_max": module.get_lr().float().max().item(),
            }
        )
    model.update_state()
    for row, module, old_state in zip(rows, modules, before):
        row["state_delta_rms"] = (
            (module.state.float() - old_state.float())
            .square()
            .mean()
            .sqrt()
            .item()
        )
        row["state_rms_after"] = (
            module.state.float().square().mean().sqrt().item()
        )
        row["effective_state_rms_after"] = (
            module.get_s().float().square().mean().sqrt().item()
        )
    return rows


def compact_updates(rows, horizon):
    selected_episodes = sorted(
        set((0, min(7, horizon - 2), min(15, horizon - 2), horizon - 2))
    )
    by_episode = {}
    for episode in selected_episodes:
        episode_rows = [
            row for row in rows if row["episode"] == episode
        ]
        by_episode[str(episode)] = {
            key: describe([row[key] for row in episode_rows])
            for key in (
                "raw_grad_rms",
                "momentum_rms",
                "state_delta_rms",
                "state_rms_after",
                "effective_state_rms_after",
                "lr_rms",
                "lr_max",
            )
        }
    by_layer = {}
    for layer in range(16):
        layer_rows = [row for row in rows if row["layer"] == layer]
        by_layer[str(layer)] = {
            key: describe([row[key] for row in layer_rows])
            for key in (
                "raw_grad_rms",
                "state_delta_rms",
                "state_rms_after",
                "effective_state_rms_after",
            )
        }
    return {"by_episode": by_episode, "by_layer": by_layer}


def run(model, batch):
    input_ids = batch["input_ids"]
    assistant_mask = batch["assistant_mask"]
    horizon = input_ids.shape[1]
    model.empty_state()
    model.zero_grad(set_to_none=False)
    losses = []
    updates = []
    for episode in range(horizon):
        episode_loss = loss(
            model,
            input_ids[:, episode],
            assistant_mask[:, episode],
        )
        episode_loss.backward()
        losses.append(episode_loss.item())
        if episode != horizon - 1:
            updates.extend(update_with_stats(model, episode))
    result = {
        "losses": losses,
        "mean_loss": sum(losses) / len(losses),
        "gradient_summary": gradient_summary(model),
        "updates": compact_updates(updates, horizon),
    }
    model.empty_state()
    model.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/dev/shm"),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["fresh", "step250"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cluster-length", type=int, default=16)
    args = parser.parse_args()
    batch = load_batch(args.batch_size, args.cluster_length)
    root = args.checkpoint_root
    config_path = root / "oloop-alpha/000000000250/config.json"
    state_path = root / "oloop-alpha/000000000250/model.pt"
    base_path = root / "llama-base/000000000000/model.pt"
    for label in args.labels:
        model = load_model(
            config_path,
            state_path,
            base_path if label == "fresh" else None,
            args.batch_size,
        )
        result = run(model, batch)
        print(json.dumps({"label": label, **result}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
