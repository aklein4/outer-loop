"""Matched-data geometry comparison for Forte-offset checkpoints.

CUDA/PyTorch only; deliberately does not import torch-xla.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator  # noqa: E402
from models import _clean_wrapped_state_dict, load_checkpoint  # noqa: E402
import models.forte as forte  # noqa: E402
from models.forte import ForteMode  # noqa: E402
from utils.torch_modules import enable_gradient_checkpointing  # noqa: E402
from utils.torch_utils import unit_softplus  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def layer_number(name: str) -> int | None:
    match = re.search(r"(backbone|output)_layers\.layers\.(\d+)", name)
    if not match:
        return None
    return int(match.group(2)) + (12 if match.group(1) == "output" else 0)


def sample_valid(x: torch.Tensor, valid: torch.Tensor, limit: int) -> torch.Tensor:
    """Take deterministic, evenly spaced valid tokens independently per trajectory."""
    x = x.detach()
    chunks = []
    for batch in range(x.shape[0]):
        indices = valid[batch].bool().nonzero(as_tuple=False).flatten()
        if not indices.numel():
            continue
        if indices.numel() > limit:
            take = torch.linspace(0, indices.numel() - 1, limit, device=indices.device).long()
            indices = indices[take]
        chunks.append(x[batch, indices].detach().to(torch.float16).cpu())
    return torch.cat(chunks, 0)


class GeometryCollector:
    def __init__(self, model, selected_episodes: set[int], tokens_per_episode: int):
        self.model = model
        self.selected = selected_episodes
        self.tokens = tokens_per_episode
        self.episode = -1
        self.valid: torch.Tensor | None = None
        self.record_hooks = False
        self.samples: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
        self.top_samples: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.matrices: dict[tuple[str, int, int], torch.Tensor] = {}
        self.last_update: dict[int, torch.Tensor] = {}
        self.layer_by_weight = {m.down_fast.weight.data_ptr(): i for i, m in enumerate(model.fast_modules())}
        self.original = forte._get_G
        self.handles = []

    def install(self) -> None:
        forte._get_G = self.get_g
        for layer, module in enumerate(self.model._causal_layers()):
            self.handles.append(module.register_forward_hook(self.activation_hook(layer)))
        for layer, module in enumerate(self.model.fast_modules()):
            self.handles.append(module.down_fast.register_forward_hook(
                self.write_hook("fast_mlp_write", layer)
            ))
            self.handles.append(module.down_proj.register_forward_hook(
                self.write_hook("standard_mlp_write", layer)
            ))

    def remove(self) -> None:
        forte._get_G = self.original
        for handle in self.handles:
            handle.remove()

    def activation_hook(self, layer: int):
        @torch.no_grad()
        def hook(_module, inputs, output):
            if not self.record_hooks or self.episode not in self.selected:
                return
            assert self.valid is not None
            self.samples[("residual_input", layer)].append(sample_valid(inputs[0], self.valid, self.tokens))
            self.samples[("residual_output", layer)].append(sample_valid(output, self.valid, self.tokens))
        return hook

    def write_hook(self, kind: str, layer: int):
        @torch.no_grad()
        def hook(_module, _inputs, output):
            if not self.record_hooks or self.episode not in self.selected:
                return
            assert self.valid is not None
            self.samples[(kind, layer)].append(sample_valid(output, self.valid, self.tokens))
        return hook

    def add_top(self, kind: str, x: torch.Tensor, valid: torch.Tensor) -> None:
        if self.episode in self.selected:
            self.top_samples[kind].append(sample_valid(x, valid, self.tokens))

    def get_g(self, activations, output_grad, down_weight, valid_mask, *dynamic_args,
              eps: float, offset_alpha: float):
        raw_g, update = self.original(activations, output_grad, down_weight, valid_mask,
                                      *dynamic_args, eps=eps, offset_alpha=offset_alpha)
        if self.episode not in self.selected:
            return raw_g, update
        layer = self.layer_by_weight[down_weight.data_ptr()]
        with torch.no_grad():
            offset, lr, offset_lr, gradient_logits, offset_logits = [x.detach().float() for x in dynamic_args]
            mask = valid_mask.bool(); maskf = mask[..., None].float()
            count = maskf.sum(-2, keepdim=True).clamp_min(1)
            a = activations.detach().float() * maskf
            g = F.linear(output_grad.detach().float(), down_weight.detach().float().T) * maskf
            a_norm = a * torch.rsqrt(a.square().sum(-2, keepdim=True) / count + eps ** 2)
            g_norm = g * torch.rsqrt(g.square().sum(-2, keepdim=True) / count + eps ** 2)
            gg, og = unit_softplus(gradient_logits), unit_softplus(offset_logits)
            base = -torch.einsum("blo,bli->boi", forte.gate_heads(g_norm, gg), a_norm) * lr

            for kind, tensor in (
                ("fast_activation", a), ("normalized_fast_activation", a_norm),
                ("fast_output_gradient", g), ("normalized_fast_output_gradient", g_norm),
                ("offset_vector", offset),
                ("gradient_gate_logits", gradient_logits), ("offset_gate_logits", offset_logits),
                ("gradient_gate", gg), ("offset_gate", og),
            ):
                self.samples[(kind, layer)].append(sample_valid(tensor, mask, self.tokens))
            # The offset branch is norm-capped before it is added.  Save the
            # realized contribution, not the pre-cap controller output.
            offset_update = update.detach().float() - base
            for kind, tensor in (
                ("raw_G", raw_g),
                ("base_update", base),
                ("offset_update", offset_update),
                ("total_update", update),
            ):
                self.matrices[(kind, layer, self.episode)] = tensor.detach().to(torch.float16).cpu()
            self.last_update[layer] = update.detach()
        return raw_g, update

    @torch.no_grad()
    def record_state(self) -> None:
        if self.episode not in self.selected:
            return
        for layer, module in enumerate(self.model.fast_modules()):
            self.matrices[("state", layer, self.episode)] = module.state.detach().to(torch.float16).cpu()
            self.matrices[("grad_buffer", layer, self.episode)] = module.grad_buffer.detach().to(torch.float16).cpu()


def loss_and_grad(model, states, ids, assistant, valid, iterations: int, aux_weight: float):
    batch = states.shape[0]
    labels = ids[:, 1:]
    assistant = assistant[:, 1:].float(); valid = valid[:, 1:].float()
    nonassistant = valid - assistant
    aw = assistant / assistant.sum(-1, keepdim=True).clamp_min(1) / batch
    nw = nonassistant / nonassistant.sum(-1, keepdim=True).clamp_min(1) / batch
    leaf = states.detach().reshape(-1, iterations, states.shape[-1]).requires_grad_(True)
    labels, aw, nw = (x.reshape(-1, iterations) for x in (labels, aw, nw))
    losses = []
    for i in range(iterations):
        logits = model.lm_head(leaf[:, i]).float()
        raw = F.cross_entropy(logits, labels[:, i].contiguous(), reduction="none")
        loss = (raw * aw[:, i]).sum() + aux_weight * (raw * nw[:, i]).sum()
        loss.backward(); losses.append(loss.detach())
    return torch.stack(losses).sum(), leaf.grad.reshape_as(states).detach().to(states.dtype)


def first_pass(model, ids, assistant, valid, collector, iterations, aux_weight):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inferred = model.forward_backbone(ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, valid).detach()
        collector.add_top("inferred_backbone", inferred, valid)
        collector.add_top("lr_embeddings", embeddings, valid)
    collector.valid = valid; collector.record_hooks = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_backbone(ids, mode=ForteMode.TRAIN_FIRST,
                                        embeddings=embeddings, embedding_mask=valid)
        states = model.forward_lm_states(hidden, mode=ForteMode.TRAIN_FIRST,
                                        logits_to_keep=slice(0, -1), embeddings=embeddings,
                                        embedding_mask=valid)
        collector.record_hooks = False
        collector.add_top("lm_states", states, valid[:, :-1])
        loss, grad = loss_and_grad(model, states, ids, assistant, valid, iterations, aux_weight)
    torch.autograd.backward(states, grad, inputs=model.grad_containers())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.update_state(ForteMode.TRAIN_FIRST)
    collector.record_state()
    return loss.item()


def run_checkpoint(step: int, repo: str, batch: dict[str, torch.Tensor], selected: set[int],
                   tokens: int, iterations: int, aux_weight: float) -> tuple[GeometryCollector, list[float]]:
    torch.cuda.empty_cache()
    model = load_checkpoint(repo, step, attention_kernel=None).cuda().eval()
    model.lm_head.to(torch.bfloat16)
    enable_gradient_checkpointing(model, True); model.train()
    gpu = {k: v.cuda() for k, v in batch.items()}
    ids, assistant, valid = (gpu[k] for k in ("input_ids", "assistant_mask", "attention_mask"))
    model.init_state(ids.shape[0], torch.device("cuda"))
    collector = GeometryCollector(model, selected, tokens); collector.install()
    losses = []
    try:
        for episode in range(ids.shape[1]):
            collector.episode = episode
            losses.append(first_pass(model, ids[:, episode], assistant[:, episode], valid[:, episode],
                                     collector, iterations, aux_weight))
            print(f"step {step} episode {episode + 1}/{ids.shape[1]} loss={losses[-1]:.5f}", flush=True)
    finally:
        collector.remove()
    del model, gpu, ids, assistant, valid
    gc.collect(); torch.cuda.empty_cache()
    return collector, losses


def centered_gram(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(0, keepdim=True)
    gram = x @ x.T / max(x.shape[1], 1)
    return gram - gram.mean(0, keepdim=True) - gram.mean(1, keepdim=True) + gram.mean()


@torch.no_grad()
def representation_geometry(a_cpu: torch.Tensor, b_cpu: torch.Tensor, pair_limit: int = 512) -> dict:
    assert a_cpu.shape == b_cpu.shape
    a, b = a_cpu.float().cuda(), b_cpu.float().cuda()
    af, bf = a.flatten(1), b.flatten(1)
    matched = F.cosine_similarity(af, bf, dim=1)
    ac, bc = af - af.mean(0, keepdim=True), bf - bf.mean(0, keepdim=True)
    ka, kb = centered_gram(af), centered_gram(bf)
    hsic = (ka * kb).sum()
    cka = hsic / (ka.square().sum().sqrt() * kb.square().sum().sqrt()).clamp_min(1e-30)
    tr_a, tr_b = ka.diag().sum(), kb.diag().sum()
    pr_a = tr_a.square() / ka.square().sum().clamp_min(1e-30)
    pr_b = tr_b.square() / kb.square().sum().clamp_min(1e-30)
    n = min(pair_limit, af.shape[0])
    idx = torch.linspace(0, af.shape[0] - 1, n, device=af.device).long()
    an, bn = F.normalize(af[idx], dim=1), F.normalize(bf[idx], dim=1)
    sa, sb = an @ an.T, bn @ bn.T
    tri = torch.triu_indices(n, n, offset=1, device=af.device)
    va, vb = sa[tri[0], tri[1]], sb[tri[0], tri[1]]
    va = va - va.mean(); vb = vb - vb.mean()
    geometry_corr = (va * vb).sum() / (va.norm() * vb.norm()).clamp_min(1e-30)
    out = {
        "samples": af.shape[0], "dimensions": af.shape[1],
        "matched_cosine_mean": matched.mean().item(), "matched_cosine_p05": torch.quantile(matched, .05).item(),
        "centered_coordinate_cosine": F.cosine_similarity(ac.flatten()[None], bc.flatten()[None]).item(),
        "linear_cka": cka.item(), "pairwise_cosine_geometry_correlation": geometry_corr.item(),
        "rms_200": af.square().mean().sqrt().item(), "rms_250": bf.square().mean().sqrt().item(),
        "rms_ratio": (bf.square().mean() / af.square().mean()).sqrt().item(),
        "relative_delta_norm": ((bf - af).norm() / af.norm().clamp_min(1e-30)).item(),
        "participation_ratio_200": pr_a.item(), "participation_ratio_250": pr_b.item(),
        "participation_ratio_change": (pr_b / pr_a).item(),
        "mean_pairwise_cosine_200": sa[tri[0], tri[1]].mean().item(),
        "mean_pairwise_cosine_250": sb[tri[0], tri[1]].mean().item(),
    }
    del a, b, af, bf, ac, bc, ka, kb, sa, sb
    return out


@torch.no_grad()
def spectral_norm_power(matrix: torch.Tensor, iterations: int = 12) -> float:
    vector = torch.randn(matrix.shape[-1], device=matrix.device, generator=torch.Generator(device=matrix.device).manual_seed(123))
    vector = F.normalize(vector, dim=0)
    for _ in range(iterations):
        left = F.normalize(matrix @ vector, dim=0)
        vector = F.normalize(matrix.T @ left, dim=0)
    return (matrix @ vector).norm().item()


@torch.no_grad()
def matrix_geometry(a_cpu: torch.Tensor, b_cpu: torch.Tensor, probe_count: int = 64) -> dict:
    assert a_cpu.shape == b_cpu.shape
    a, b = a_cpu.float().cuda(), b_cpu.float().cuda()
    batch, out_dim, in_dim = a.shape
    af, bf = a.flatten(1), b.flatten(1)
    matched = F.cosine_similarity(af, bf, dim=1)
    gen = torch.Generator(device="cuda").manual_seed(711)
    pin = F.normalize(torch.randn(probe_count, in_dim, device="cuda", generator=gen), dim=1)
    pout = F.normalize(torch.randn(probe_count, out_dim, device="cuda", generator=gen), dim=1)
    ya, yb = torch.einsum("ki,boi->bko", pin, a), torch.einsum("ki,boi->bko", pin, b)
    za, zb = torch.einsum("ko,boi->bki", pout, a), torch.einsum("ko,boi->bki", pout, b)
    action_cos = F.cosine_similarity(ya, yb, dim=-1).mean()
    transpose_cos = F.cosine_similarity(za, zb, dim=-1).mean()
    ma, mb = a.mean(0), b.mean(0)
    sigma_a, sigma_b = spectral_norm_power(ma), spectral_norm_power(mb)
    stable_a = ma.square().sum().item() / max(sigma_a ** 2, 1e-30)
    stable_b = mb.square().sum().item() / max(sigma_b ** 2, 1e-30)
    out = {
        "batch": batch, "output_dim": out_dim, "input_dim": in_dim,
        "matrix_cosine": matched.mean().item(), "matrix_cosine_min": matched.min().item(),
        "relative_delta_norm": ((bf - af).norm(dim=1) / af.norm(dim=1).clamp_min(1e-30)).mean().item(),
        "norm_ratio": (bf.norm(dim=1) / af.norm(dim=1).clamp_min(1e-30)).mean().item(),
        "input_action_cosine": action_cos.item(), "transpose_action_cosine": transpose_cos.item(),
        "stable_rank_200": stable_a, "stable_rank_250": stable_b,
        "stable_rank_ratio": stable_b / max(stable_a, 1e-30),
    }
    del a, b, af, bf, ya, yb, za, zb, ma, mb
    return out


def parameter_family(name: str) -> str | None:
    ordered = (
        ("offset_log_lr", "offset_log_lr"), (".log_lr", "log_lr"),
        ("gradient_gate_proj", "gradient_gate_proj"), ("offset_gate_proj", "offset_gate_proj"),
        ("offset_proj", "offset_proj"), ("up_fast", "up_fast"), ("gate_fast", "gate_fast"),
        ("down_fast", "down_fast"), ("self_attn", "self_attention"),
        (".mlp.up_proj", "up_proj"), (".mlp.gate_proj", "gate_proj"), (".mlp.down_proj", "down_proj"),
    )
    for pattern, family in ordered:
        if pattern in name:
            return family
    return None


@torch.no_grad()
def parameter_geometry(root: Path, output_dir: Path) -> list[dict]:
    states = {}
    for step in (200, 250):
        states[step] = _clean_wrapped_state_dict(torch.load(root / f"{step:012d}" / "model.pt",
                                                             map_location="cpu", weights_only=True))
    rows = []
    for name in sorted(set(states[200]) & set(states[250])):
        family, layer = parameter_family(name), layer_number(name)
        if family is None or layer is None or states[200][name].ndim != 2:
            continue
        a, b = states[200][name].float().cuda(), states[250][name].float().cuda()
        if a.shape != b.shape:
            continue
        af, bf = a.flatten(), b.flatten()
        gen = torch.Generator(device="cuda").manual_seed(1000 + layer * 37 + len(name))
        probes_in = F.normalize(torch.randn(64, a.shape[1], device="cuda", generator=gen), dim=1)
        probes_out = F.normalize(torch.randn(64, a.shape[0], device="cuda", generator=gen), dim=1)
        ya, yb = probes_in @ a.T, probes_in @ b.T
        za, zb = probes_out @ a, probes_out @ b
        sigma_a, sigma_b = spectral_norm_power(a), spectral_norm_power(b)
        row = {
            "name": name, "family": family, "layer": layer,
            "output_dim": a.shape[0], "input_dim": a.shape[1], "elements": a.numel(),
            "weight_cosine": F.cosine_similarity(af[None], bf[None]).item(),
            "relative_delta_norm": ((bf - af).norm() / af.norm().clamp_min(1e-30)).item(),
            "norm_ratio": (bf.norm() / af.norm().clamp_min(1e-30)).item(),
            "input_action_cosine": F.cosine_similarity(ya, yb, dim=1).mean().item(),
            "transpose_action_cosine": F.cosine_similarity(za, zb, dim=1).mean().item(),
            "stable_rank_200": a.square().sum().item() / max(sigma_a ** 2, 1e-30),
            "stable_rank_250": b.square().sum().item() / max(sigma_b ** 2, 1e-30),
        }
        row["stable_rank_ratio"] = row["stable_rank_250"] / max(row["stable_rank_200"], 1e-30)
        rows.append(row)
        del a, b, af, bf, ya, yb, za, zb
        if len(rows) % 25 == 0:
            print(f"parameter geometry {len(rows)} matrices", flush=True)
    write_csv(output_dir / "parameter_function_geometry.csv", rows)
    del states
    gc.collect(); torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="aklein4/horizon-v2_forte-offset")
    ap.add_argument("--trajectories", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--selected-episodes", type=int, nargs="+", default=[0, 7, 15, 31, 63])
    ap.add_argument("--tokens-per-episode", type=int, default=64)
    ap.add_argument("--num-logit-iterations", type=int, default=4)
    ap.add_argument("--aux-loss-weight", type=float, default=.1)
    ap.add_argument("--data-config", type=Path, default=SRC / "configs/data/300k-horizons-llama3.yaml")
    ap.add_argument("--output-dir", type=Path,
                    default=SRC / "local_data/horizon_v2_forte_offset_spike_analysis/geometry")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(42); np.random.seed(42); torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.data_config)
    iterator = iter(datasets.load_dataset(cfg.dataset.url, **cfg.dataset.kwargs))
    raw = [next(iterator) for _ in range(args.trajectories)]
    kwargs = OmegaConf.to_container(cfg.collator.kwargs, resolve=True); kwargs["cluster_length"] = args.episodes
    collated = HorizonCollator(**kwargs)(raw)
    batch = {k: collated[k].cpu() for k in ("input_ids", "assistant_mask", "attention_mask")}
    selected = set(args.selected_episodes)

    captures, losses = {}, {}
    for step in (200, 250):
        captures[step], losses[step] = run_checkpoint(step, args.repo, batch, selected,
                                                      args.tokens_per_episode, args.num_logit_iterations,
                                                      args.aux_loss_weight)

    representation_rows = []
    sample_keys = sorted(set(captures[200].samples) & set(captures[250].samples))
    for index, (kind, layer) in enumerate(sample_keys):
        a = torch.cat(captures[200].samples[(kind, layer)], 0)
        b = torch.cat(captures[250].samples[(kind, layer)], 0)
        representation_rows.append({"kind": kind, "layer": layer, **representation_geometry(a, b)})
        if (index + 1) % 20 == 0:
            print(f"representation geometry {index + 1}/{len(sample_keys)}", flush=True)
    for kind in sorted(set(captures[200].top_samples) & set(captures[250].top_samples)):
        a = torch.cat(captures[200].top_samples[kind], 0)
        b = torch.cat(captures[250].top_samples[kind], 0)
        representation_rows.append({"kind": kind, "layer": "", **representation_geometry(a, b)})
    write_csv(args.output_dir / "representation_geometry.csv", representation_rows)

    matrix_rows = []
    matrix_keys = sorted(set(captures[200].matrices) & set(captures[250].matrices))
    for index, (kind, layer, episode) in enumerate(matrix_keys):
        matrix_rows.append({"kind": kind, "layer": layer, "episode": episode,
                            **matrix_geometry(captures[200].matrices[(kind, layer, episode)],
                                              captures[250].matrices[(kind, layer, episode)])})
        if (index + 1) % 40 == 0:
            print(f"matrix geometry {index + 1}/{len(matrix_keys)}", flush=True)
    write_csv(args.output_dir / "recurrent_matrix_geometry.csv", matrix_rows)

    del captures
    gc.collect(); torch.cuda.empty_cache()
    checkpoint_root = SRC / "local_data/checkpoints" / args.repo.replace("/", "--")
    parameter_rows = parameter_geometry(checkpoint_root, args.output_dir)

    metadata = {
        "repo": args.repo, "steps": [200, 250], "dataset": cfg.dataset.url,
        "trajectories": args.trajectories, "episodes": args.episodes,
        "selected_episodes": args.selected_episodes, "tokens_per_episode_per_trajectory": args.tokens_per_episode,
        "valid_tokens": int(batch["attention_mask"].sum()), "assistant_tokens": int(batch["assistant_mask"].sum()),
        "sources": [row.get("source") for row in raw], "loss_mean": {str(s): float(np.mean(v)) for s, v in losses.items()},
        "representation_rows": len(representation_rows), "matrix_rows": len(matrix_rows),
        "parameter_rows": len(parameter_rows), "torch_xla_used": False,
        "device": torch.cuda.get_device_name(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
