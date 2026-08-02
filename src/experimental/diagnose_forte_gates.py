"""CUDA diagnostic for Forte fast-weight gates and two-pass training math."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import datasets
import torch
from omegaconf import OmegaConf


SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator
from models import load_checkpoint
import models.forte as forte_module
from models.forte import ForteMode


def summary(x: torch.Tensor, sample_limit: int = 1_000_000) -> dict:
    x = x.detach().float().reshape(-1)
    finite = torch.isfinite(x)
    out = {
        "count": x.numel(),
        "finite_fraction": finite.sum().item() / x.numel() if x.numel() else 1.0,
    }
    x = x[finite]
    if not x.numel():
        return out
    if x.numel() > sample_limit:
        step = math.ceil(x.numel() / sample_limit)
        qx = x[::step]
    else:
        qx = x
    qs = torch.quantile(
        qx,
        torch.tensor([0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1], device=x.device),
    ).cpu().tolist()
    out.update({
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "rms": x.square().mean().sqrt().item(),
        "zero_fraction": (x == 0).float().mean().item(),
        "quantiles": dict(zip(
            ["0", ".001", ".01", ".05", ".25", ".5", ".75", ".95", ".99", ".999", "1"],
            qs,
        )),
    })
    return out


def corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    return (x @ y / denom.clamp_min(1e-30)).item()


class Collector:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self.phase = ""
        self.episode = -1
        self.records = []
        self.activation_records = []
        self.state_records = []
        self.layer_by_weight = {
            mlp.down_fast.weight.data_ptr(): i
            for i, mlp in enumerate(model.fast_modules())
        }
        self.original_get_g = forte_module._get_G
        self.handles = []

    def install(self):
        forte_module._get_G = self.get_g
        for i, layer in enumerate(self.model._causal_layers()):
            self.handles.append(layer.register_forward_hook(self.activation_hook(i)))

    def remove(self):
        forte_module._get_G = self.original_get_g
        for h in self.handles:
            h.remove()

    def activation_hook(self, layer):
        def hook(_module, inputs, output):
            if not self.phase.startswith("first"):
                return
            x = inputs[0]
            mask = self.masks[self.episode].to(x.device)[None]
            vals_in = x[mask]
            vals_out = output[mask]
            self.activation_records.append({
                "episode": self.episode,
                "layer": layer,
                "input": summary(vals_in),
                "output": summary(vals_out),
            })
        return hook

    def get_g(self, activations, output_grad, down_weight,
              activation_gate_logits, gradient_gate_logits, valid_mask, eps):
        G, update = self.original_get_g(
            activations, output_grad, down_weight,
            activation_gate_logits, gradient_gate_logits, valid_mask, eps,
        )
        layer = self.layer_by_weight[down_weight.data_ptr()]
        mask = self.masks[self.episode].to(activations.device)[None]
        al = activation_gate_logits.float()[mask]
        gl = gradient_gate_logits.float()[mask]
        ag = 2 * torch.sigmoid(al)
        gg = 2 * torch.sigmoid(gl)
        a = activations.float()
        g = torch.nn.functional.linear(output_grad.float(), down_weight.T.float())
        a_norm = a * torch.rsqrt(a.square().mean(dim=-2, keepdim=True) + eps)
        g_norm = g * torch.rsqrt(g.square().sum(dim=-2, keepdim=True) + eps)
        un = G.float().reshape(-1)
        up = update.float().reshape(-1)
        denom = un.norm() * up.norm()
        channel_a = ag.mean(0)
        channel_g = gg.mean(0)
        self.records.append({
            "phase": self.phase,
            "episode": self.episode,
            "layer": layer,
            "activation_logits": summary(al),
            "gradient_logits": summary(gl),
            "activation_gate_2sigmoid": summary(ag),
            "gradient_gate_2sigmoid": summary(gg),
            "activation_gate_saturation_lt_0.1": (ag < .1).float().mean().item(),
            "activation_gate_saturation_gt_1.9": (ag > 1.9).float().mean().item(),
            "gradient_gate_saturation_lt_0.1": (gg < .1).float().mean().item(),
            "gradient_gate_saturation_gt_1.9": (gg > 1.9).float().mean().item(),
            "activation_gradient_gate_correlation": corr(ag, gg),
            "activation_gate_channel_mean": summary(channel_a),
            "gradient_gate_channel_mean": summary(channel_g),
            "activation_gate_top_channels": torch.topk(channel_a, 8).indices.cpu().tolist(),
            "activation_gate_bottom_channels": torch.topk(channel_a, 8, largest=False).indices.cpu().tolist(),
            "gradient_gate_top_channels": torch.topk(channel_g, 8).indices.cpu().tolist(),
            "gradient_gate_bottom_channels": torch.topk(channel_g, 8, largest=False).indices.cpu().tolist(),
            "activation_normed": summary(a_norm[mask]),
            "gradient_normed": summary(g_norm[mask]),
            "raw_G_norm": G.float().norm().item(),
            "gated_update_norm": update.float().norm().item(),
            "update_to_G_norm_ratio": (update.float().norm() / G.float().norm().clamp_min(1e-30)).item(),
            "update_G_cosine": ((un @ up) / denom.clamp_min(1e-30)).item(),
            "G": summary(G),
            "update": summary(update),
        })
        return G, update

    @torch.no_grad()
    def record_state(self):
        for layer, mlp in enumerate(self.model.fast_modules()):
            self.state_records.append({
                "phase": self.phase,
                "episode": self.episode,
                "layer": layer,
                "state_norm": mlp.state.float().norm().item(),
                "state_max_abs": mlp.state.float().abs().max().item(),
                "grad_buffer_norm": mlp.grad_buffer.float().norm().item(),
                "finite": bool(torch.isfinite(mlp.state).all() and torch.isfinite(mlp.grad_buffer).all()),
            })


def loss_and_lm_grad(model, lm_states, input_ids, assistant_mask, iterations=1):
    labels = input_ids[:, 1:]
    output_mask = assistant_mask[:, 1:].float()
    weights = output_mask / output_mask.sum(-1, keepdim=True).clamp_min(1) / lm_states.shape[0]
    leaf = lm_states.detach().reshape(-1, iterations, lm_states.shape[-1]).requires_grad_(True)
    labels = labels.reshape(-1, iterations)
    weights = weights.reshape(-1, iterations)
    losses = []
    for i in range(iterations):
        logits = model.lm_head(leaf[:, i]).float()
        loss = torch.nn.functional.cross_entropy(logits, labels[:, i], reduction="none")
        loss = (loss * weights[:, i]).sum()
        losses.append(loss.detach())
        loss.backward()
    return torch.stack(losses).sum(), leaf.grad.reshape_as(lm_states).detach().to(lm_states.dtype)


def first_pass(model, ids, assistant, mask, collector, final=False):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_backbone(ids, mode=ForteMode.TRAIN_FIRST,
                                        embeddings=embeddings, embedding_mask=mask)
        states = model.forward_lm_states(hidden, mode=ForteMode.TRAIN_FIRST,
                                        logits_to_keep=slice(0, -1), embeddings=embeddings,
                                        embedding_mask=mask)
        loss, grad = loss_and_lm_grad(model, states, ids, assistant)
    if final:
        torch.autograd.backward(states, grad)
    else:
        torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(embeddings, mask,
                           ForteMode.TRAIN_SECOND if final else ForteMode.TRAIN_FIRST)
    collector.record_state()
    return loss.item()


def second_pass(model, ids, assistant, mask, collector):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, mask)
        hidden = model.forward_backbone(ids, mode=ForteMode.TRAIN_SECOND,
                                        embeddings=embeddings, embedding_mask=mask,
                                        future_loss_scale=1.)
        states = model.forward_lm_states(hidden, mode=ForteMode.TRAIN_SECOND,
                                        logits_to_keep=slice(0, -1), embeddings=embeddings,
                                        embedding_mask=mask, future_loss_scale=1.)
        loss, grad = loss_and_lm_grad(model, states, ids, assistant)
    torch.autograd.backward(states, grad)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(embeddings, mask, ForteMode.TRAIN_SECOND)
    collector.record_state()
    return loss.item()


def parameter_report(model):
    groups = defaultdict(list)
    irregular = []
    for name, p in model.named_parameters():
        s = summary(p)
        group = "other"
        for key in ("activation_gate_proj", "gradient_gate_proj", "log_lr",
                    "up_fast", "gate_fast", "down_fast", "embedding_state"):
            if key in name:
                group = key
                break
        groups[group].append({"name": name, **s})
        if s["finite_fraction"] < 1 or s.get("rms", 0) == 0 or s.get("quantiles", {}).get("1", 0) > 1e3:
            irregular.append({"name": name, **s})
    compact = {}
    for group, rows in groups.items():
        compact[group] = {
            "tensors": len(rows),
            "elements": sum(r["count"] for r in rows),
            "rms_across_tensors": summary(torch.tensor([r["rms"] for r in rows])),
            "max_abs_across_tensors": max(max(abs(r["quantiles"]["0"]), abs(r["quantiles"]["1"])) for r in rows),
        }
    return compact, irregular


def gradient_report(model):
    rows, missing = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(name)
        else:
            s = summary(p.grad)
            rows.append({"name": name, **s})
    return {
        "missing_count": len(missing),
        "missing": missing,
        "finite_fraction": sum(r["finite_fraction"] * r["count"] for r in rows) / max(1, sum(r["count"] for r in rows)),
        "global_norm": math.sqrt(sum((r["rms"] ** 2) * r["count"] for r in rows)),
        "largest_tensor_norms": sorted(
            [{"name": r["name"], "norm": r["rms"] * math.sqrt(r["count"]), "max_abs": max(abs(r["quantiles"]["0"]), abs(r["quantiles"]["1"]))} for r in rows],
            key=lambda z: z["norm"], reverse=True,
        )[:30],
        "nonfinite": [r["name"] for r in rows if r["finite_fraction"] < 1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="aklein4/Horizon-TPU_forte-1b")
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("forte_gate_diagnostics_step50.json"))
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    model = load_checkpoint(args.repo, args.step, attention_kernel=None).cuda().eval()
    model.lm_head.to(dtype=torch.bfloat16)
    params, param_irregular = parameter_report(model)

    cfg = OmegaConf.load(SRC / "configs/data/horizons-llama3.yaml")
    raw_ds = datasets.load_dataset(cfg.dataset.url, **cfg.dataset.kwargs)
    raw = next(iter(raw_ds))
    collator_kwargs = OmegaConf.to_container(cfg.collator.kwargs, resolve=True)
    collator_kwargs["cluster_length"] = args.episodes
    batch = HorizonCollator(**collator_kwargs)([raw])
    ids = batch["input_ids"][0].cuda()
    assistant = batch["assistant_mask"][0].cuda()
    masks = batch["attention_mask"][0].cuda()
    data_report = {
        "source": raw.get("source"),
        "cluster_num_tokens": raw.get("num_tokens"),
        "average_similarity_to_mean": raw.get("average_similarity_to_mean"),
        "episodes": args.episodes,
        "nonpad_tokens": masks.sum(-1).cpu().tolist(),
        "assistant_tokens": assistant.sum(-1).cpu().tolist(),
    }

    model.init_state(1, torch.device("cuda"))
    collector = Collector(model, masks)
    collector.install()
    first_losses, second_losses = [], []
    try:
        for i in range(args.episodes):
            collector.phase, collector.episode = "first", i
            first_losses.append(first_pass(model, ids[i:i+1], assistant[i:i+1], masks[i:i+1], collector))
        model.finalize_state()
        final_grad_norms = [m.final_grad_norm.item() for m in model.fast_modules()]
        model.zero_grad(set_to_none=False)
        for i in range(args.episodes - 1):
            collector.phase, collector.episode = "second", i
            second_losses.append(second_pass(model, ids[i:i+1], assistant[i:i+1], masks[i:i+1], collector))
        collector.phase, collector.episode = "second_final", args.episodes - 1
        second_losses.append(first_pass(model, ids[-1:], assistant[-1:], masks[-1:], collector, final=True))
        relative_grad_error = model.relative_grad_error().item()
        grads = gradient_report(model)
    finally:
        collector.remove()

    dynamic_lrs = []
    with torch.no_grad():
        for i, mlp in enumerate(model.fast_modules()):
            lr = torch.exp(mlp.fast_dynamic_lr.log_lr.float() * math.sqrt(mlp.fast_weight_size)
                           + math.log(mlp.fast_dynamic_lr.base_lr)
                           - math.log(mlp.fast_weight_size))
            dynamic_lrs.append({"layer": i, **summary(lr)})
    result = {
        "checkpoint": {"repo": args.repo, "step": args.step},
        "device": torch.cuda.get_device_name(),
        "data": data_report,
        "losses": {"first": first_losses, "second": second_losses},
        "parameter_groups": params,
        "parameter_irregularities": param_irregular,
        "dynamic_learning_rates": dynamic_lrs,
        "gate_records": collector.records,
        "activation_records": collector.activation_records,
        "state_records": collector.state_records,
        "final_grad_buffer_norms": final_grad_norms,
        "relative_grad_error": relative_grad_error,
        "gradients": grads,
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "data": data_report,
        "losses": result["losses"],
        "gate_calls": len(collector.records),
        "relative_grad_error": relative_grad_error,
        "gradient_global_norm": grads["global_norm"],
        "gradient_nonfinite": grads["nonfinite"],
        "peak_gb": result["cuda_peak_allocated_gb"],
    }, indent=2), flush=True)
    # Streaming datasets can leave an Arrow worker in teardown in some builds.
    os._exit(0)


if __name__ == "__main__":
    main()
