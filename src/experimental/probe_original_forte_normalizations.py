"""Probe normalization candidates using the exact pre-normalization Forte code.

Run with FORTE_ORIGINAL_SRC pointing at a worktree of commit 6d7726a.  This
keeps an evolving working tree from changing checkpoint behavior.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ORIGINAL_SRC = Path(os.environ.get(
    "FORTE_ORIGINAL_SRC", "/tmp/forte-original-6d7726a/src"
)).resolve()
sys.path.insert(0, str(ORIGINAL_SRC))

import datasets
import torch
from omegaconf import OmegaConf

from collators.horizon import HorizonCollator
from models import load_checkpoint
import models.forte as forte_module
from models.forte import ForteMode
from utils import constants
from utils.torch_modules import enable_gradient_checkpointing


def loss_and_lm_grad(model, states, input_ids, assistant_mask):
    labels = input_ids[:, 1:]
    weights = assistant_mask[:, 1:].float()
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1) / states.shape[0]
    leaf = states.detach().requires_grad_(True)
    logits = model.lm_head(leaf).float()
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).reshape_as(labels)
    loss = (loss * weights).sum()
    loss.backward()
    return loss.detach(), leaf.grad.detach().to(states.dtype)


def batch_norm(x):
    return x.float().flatten(1).norm(dim=1)


def batch_cosine(x, y):
    xf = x.float().flatten(1)
    yf = y.float().flatten(1)
    return (xf * yf).sum(1) / (xf.norm(dim=1) * yf.norm(dim=1)).clamp_min(1e-30)


class Collector:
    def __init__(self, model):
        self.original = forte_module._get_G
        self.position = -1
        self.rows = []
        self.layer_by_weight = {
            module.down_fast.weight.data_ptr(): layer
            for layer, module in enumerate(model.fast_modules())
        }

    def install(self):
        forte_module._get_G = self.get_g

    def remove(self):
        forte_module._get_G = self.original

    def get_g(self, activations, output_grad, down_weight,
              activation_gate_logits, gradient_gate_logits, eps):
        raw, target = self.original(
            activations, output_grad, down_weight,
            activation_gate_logits, gradient_gate_logits, eps,
        )
        a = activations.float()
        g = torch.nn.functional.linear(output_grad.float(), down_weight.T.float())
        # In the original implementation activations were already zeroed by the
        # attention mask. Probe data confirmed that invalid-position g energy is
        # also exactly zero. Infer the same mask without consulting changed code.
        valid = a.abs().sum(dim=-1, keepdim=True) != 0
        a = a * valid
        gv = g * valid
        n = valid.sum(dim=-2, keepdim=True).clamp_min(1).float()
        length = a.shape[-2]

        def sequence_norm(x, denominator):
            energy = x.square().sum(dim=-2, keepdim=True)
            if denominator == "none":
                return x
            if denominator == "padded_mean":
                return x * torch.rsqrt(energy / length + eps)
            if denominator == "valid_mean":
                return x * torch.rsqrt(energy / n + eps)
            if denominator == "sum":
                return x * torch.rsqrt(energy + eps)
            raise ValueError(denominator)

        a_pad = sequence_norm(a, "padded_mean")
        a_valid = sequence_norm(a, "valid_mean")
        a_sum = sequence_norm(a, "sum")
        g_sum = sequence_norm(gv, "sum")
        g_valid = sequence_norm(gv, "valid_mean")

        def feature_rms(x):
            return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)

        def joint_rms(x):
            energy = x.square().sum(dim=(-2, -1), keepdim=True)
            count = n * x.shape[-1]
            return x * torch.rsqrt(energy / count + eps)

        def joint_l2(x):
            return x * torch.rsqrt(
                x.square().sum(dim=(-2, -1), keepdim=True) + eps
            )

        a_feature = feature_rms(a)
        g_feature = feature_rms(gv)
        a_joint = joint_rms(a)
        g_joint = joint_rms(gv)
        a_joint_l2 = joint_l2(a)
        g_joint_l2 = joint_l2(gv)

        candidates = {
            "raw": raw,
            "old_current": g_sum.mT @ a_pad,
            "masked_mean_a": g_sum.mT @ a_valid,
            "masked_mean_g": g_valid.mT @ a_pad,
            "masked_mean_both": g_valid.mT @ a_valid,
            "sum_both": g_sum.mT @ a_sum,
            "g_sum_only": g_sum.mT @ a,
            "a_masked_mean_only": gv.mT @ a_valid,
            "g_masked_mean_only": g_valid.mT @ a,
            "feature_mean_both": g_feature.mT @ a_feature,
            "feature_mean_a_only": gv.mT @ a_feature,
            "feature_mean_g_only": g_feature.mT @ a,
            "joint_mean_both": g_joint.mT @ a_joint,
            "joint_mean_a_only": gv.mT @ a_joint,
            "joint_mean_g_only": g_joint.mT @ a,
            "joint_l2_both": g_joint_l2.mT @ a_joint_l2,
            "a_sequence_g_feature": g_feature.mT @ a_valid,
            "a_sequence_g_joint": g_joint.mT @ a_valid,
            "a_feature_g_sequence": g_valid.mT @ a_feature,
            "a_joint_g_sequence": g_valid.mT @ a_joint,
        }
        # Final-matrix normalization tests whether the learned gates merely
        # normalize the already-formed update magnitude.
        for source in ("raw", "old_current", "masked_mean_both"):
            matrix = candidates[source]
            rms = matrix.square().mean(dim=(-2, -1), keepdim=True).sqrt()
            candidates[f"matrix_rms_{source}"] = matrix / rms.clamp_min(eps)

        target_norm = batch_norm(target)
        layer = self.layer_by_weight[down_weight.data_ptr()]
        for batch_index in range(a.shape[0]):
            row = {
                "trajectory": batch_index,
                "position": self.position,
                "layer": layer,
                "valid_tokens": int(n[batch_index].item()),
                "target_norm": target_norm[batch_index].item(),
                "raw_norm": batch_norm(raw)[batch_index].item(),
                "projected_gradient_norm": batch_norm(gv)[batch_index].item(),
                "activation_norm": batch_norm(a)[batch_index].item(),
            }
            for name, matrix in candidates.items():
                row[f"{name}_norm"] = batch_norm(matrix)[batch_index].item()
                row[f"{name}_cos_target"] = batch_cosine(
                    matrix, target
                )[batch_index].item()
            self.rows.append(row)
        return raw, target


def first_pass(model, ids, assistant, mask, collector):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_backbone(
            ids, mode=ForteMode.TRAIN_FIRST,
            embeddings=embeddings, embedding_mask=mask,
        )
        states = model.forward_lm_states(
            hidden, mode=ForteMode.TRAIN_FIRST, logits_to_keep=slice(0, -1),
            embeddings=embeddings, embedding_mask=mask,
        )
        loss, grad = loss_and_lm_grad(model, states, ids, assistant)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(embeddings, mask, ForteMode.TRAIN_FIRST)
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    assert os.popen(f"git -C {ORIGINAL_SRC.parent} rev-parse --short HEAD").read().strip() == "6d7726a"
    constants.CHECKPOINTS_PATH = args.checkpoint_root.resolve()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    model = load_checkpoint(
        "aklein4/Horizon-TPU_forte-1b", args.step, attention_kernel=None
    ).cuda().eval()
    model.lm_head.to(dtype=torch.bfloat16)
    enable_gradient_checkpointing(model, True)
    model.train()

    config = OmegaConf.load(ORIGINAL_SRC / "configs/data/horizons-llama3.yaml")
    stream = datasets.load_dataset(config.dataset.url, **config.dataset.kwargs)
    iterator = iter(stream)
    raw_examples = [next(iterator) for _ in range(args.trajectories)]
    batch = HorizonCollator(**OmegaConf.to_container(
        config.collator.kwargs, resolve=True
    ))(raw_examples)
    ids = batch["input_ids"].cuda()
    assistant = batch["assistant_mask"].cuda()
    masks = batch["attention_mask"].cuda()
    model.init_state(args.trajectories, torch.device("cuda"))

    collector = Collector(model)
    collector.install()
    try:
        for position in range(64):
            collector.position = position
            loss = first_pass(
                model, ids[:, position], assistant[:, position],
                masks[:, position], collector,
            )
            print(f"position {position:02d}: loss={loss:.6f}", flush=True)
    finally:
        collector.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(collector.rows[0]))
        writer.writeheader()
        writer.writerows(collector.rows)
    print(f"wrote {args.output} ({len(collector.rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
