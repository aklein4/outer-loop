"""Create self-contained per-token Piano gate and prediction visualizations.

Examples
--------
One episode from a distinct trajectory in every latent-compilation subset::

    python scripts/visualize_piano_token_gates.py sampling.n=1

Four episodes per selected subset, evenly spaced over horizon position::

    python scripts/visualize_piano_token_gates.py \
      sampling.n=4 \
      'sampling.subsets=[code-search-net--code_search_net,blitt--SPoRC]' \
      name=code_and_sporc \
      output.note='Comparison after the data refresh.'

Two complete trajectories per selected subset, one HTML per trajectory::

    python scripts/visualize_piano_token_gates.py \
      sampling.mode=trajectories sampling.n=2 \
      'sampling.subsets=[code-search-net--code_search_net]' \
      name=two_code_trajectories

Specific latent trajectories can be selected directly; this automatically uses
complete-trajectory mode::

    python scripts/visualize_piano_token_gates.py \
      'sampling.subsets=[code-search-net--code_search_net]' \
      'sampling.latents=[desertbit/glue,kubernetes/kubernetes]' \
      name=selected_code_repositories

Hydra composes ``config.model`` just as it does for training. Override a model
group or individual checkpoint fields in the usual way, for example::

    python scripts/visualize_piano_token_gates.py \
      model=old-piano-melted-step350 \
      model.pretrained_url=aklein4/horizon-v2_piano-melted \
      model.pretrained_step=350
"""
from __future__ import annotations

import gc
import html
import importlib
import json
import re
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator
from models import load_checkpoint_state
from utils import constants
from utils.import_utils import import_model
from utils.torch_modules import enable_gradient_checkpointing
from utils.torch_utils import unit_softplus


@dataclass
class Trajectory:
    uid: int
    subset: str
    source: str
    latent: str
    dataset_index: int
    row: dict[str, Any]
    target_episode: int | None = None


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def filename_stem(value: Any) -> str:
    """Validate a Hydra-provided filename stem without changing its spelling."""

    name = str(value).strip()
    if name.lower().endswith(".html"):
        name = name[:-5]
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError(
            "name must be a non-empty filename stem without directory separators"
        )
    return name


def model_url_folder(model_config: DictConfig) -> str:
    """Match evaluation output naming by replacing URL slashes with '--'."""

    if model_config.pretrained_url is not None:
        return str(model_config.pretrained_url).replace("/", "--")
    return "fresh--" + str(model_config.type).replace(".", "--")


def decoded_pieces(tokenizer, token_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids
        ],
        dtype=str,
    )


def token_label(piece: Any) -> str:
    return (
        str(piece)
        .lstrip(" ")
        .replace(" ", "␠")
        .replace("\n", "↵")
        .replace("\t", "⇥")
    )


def autocast_context(device: torch.device, dtype_name: str):
    if device.type == "cpu" or dtype_name == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype_name))


def model_mode(model, name: str):
    module = importlib.import_module(model.__class__.__module__)
    return getattr(module, "PianoMode")[name]


class GateCollector:
    """Capture post-softplus scalar token gates during TRAIN_FIRST forwards."""

    def __init__(self, model, record_positions: list[set[int]]):
        self.position = -1
        self.record_positions = record_positions
        self.gates: dict[tuple[int, int, int], np.ndarray] = {}
        self.seen: set[tuple[int, int]] = set()
        self.handles = [
            module.fast_dynamic_lr.register_forward_hook(self._hook(layer))
            for layer, module in enumerate(model.fast_modules())
        ]
        self.num_layers = len(self.handles)

    def _hook(self, layer: int):
        @torch.no_grad()
        def record(_module, inputs, output):
            key = (self.position, layer)
            if key in self.seen:
                return
            self.seen.add(key)
            gate = unit_softplus(output[1].float()).squeeze(-1)
            valid = inputs[1].bool()
            for batch_index, positions in enumerate(self.record_positions):
                if self.position not in positions:
                    continue
                length = int(valid[batch_index].sum())
                self.gates[(batch_index, self.position, layer)] = (
                    gate[batch_index, : max(0, length - 1)]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        return record

    def get(self, batch_index: int, position: int) -> np.ndarray:
        return np.stack(
            [
                self.gates.pop((batch_index, position, layer))
                for layer in range(self.num_layers)
            ]
        )

    def remove(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def empty_fast_weights(model):
    """Temporarily replace recurrent fast weights with broadcastable zeros."""

    modules = list(model.fast_modules())
    original_states = [module.state for module in modules]
    try:
        for module, state in zip(modules, original_states):
            module.state = torch.zeros_like(state[:1])
        yield
    finally:
        for module, state in zip(modules, original_states):
            module.state = state


@torch.no_grad()
def prediction_statistics(
    model,
    updated_states: torch.Tensor,
    empty_states: torch.Tensor,
    labels: torch.Tensor,
    valid_targets: torch.Tensor,
    block_size: int,
) -> dict[str, torch.Tensor]:
    """Compute label likelihood/rank and KL(updated || empty) on valid tokens."""

    batch_size, sequence_length, hidden_size = updated_states.shape
    flat_valid = valid_targets.reshape(-1)
    valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
    flat_updated = updated_states.detach().reshape(-1, hidden_size)
    flat_empty = empty_states.detach().reshape(-1, hidden_size)
    flat_labels = labels.reshape(-1)

    float_outputs = {
        name: torch.full(
            (batch_size * sequence_length,),
            float("nan"),
            device=updated_states.device,
            dtype=torch.float32,
        )
        for name in ("kl", "logp_updated", "logp_empty")
    }
    rank_outputs = {
        name: torch.zeros(
            batch_size * sequence_length,
            device=updated_states.device,
            dtype=torch.int32,
        )
        for name in ("rank_updated", "rank_empty")
    }

    for start in range(0, len(valid_indices), block_size):
        indices = valid_indices[start : start + block_size]
        block_labels = flat_labels[indices]
        logits_updated = model.lm_head(flat_updated[indices]).float()
        logits_empty = model.lm_head(flat_empty[indices]).float()
        log_z_updated = torch.logsumexp(logits_updated, dim=-1)
        log_z_empty = torch.logsumexp(logits_empty, dim=-1)
        label_updated = logits_updated.gather(1, block_labels[:, None]).squeeze(1)
        label_empty = logits_empty.gather(1, block_labels[:, None]).squeeze(1)
        probabilities_updated = torch.softmax(logits_updated, dim=-1)
        kl = (
            (probabilities_updated * (logits_updated - logits_empty)).sum(-1)
            - log_z_updated
            + log_z_empty
        ).clamp_min(0.0)

        float_outputs["kl"][indices] = kl
        float_outputs["logp_updated"][indices] = label_updated - log_z_updated
        float_outputs["logp_empty"][indices] = label_empty - log_z_empty
        rank_outputs["rank_updated"][indices] = 1 + (
            logits_updated > label_updated[:, None]
        ).sum(-1).to(torch.int32)
        rank_outputs["rank_empty"][indices] = 1 + (
            logits_empty > label_empty[:, None]
        ).sum(-1).to(torch.int32)
        del logits_updated, logits_empty, probabilities_updated

    return {
        **{
            name: value.reshape(batch_size, sequence_length)
            for name, value in float_outputs.items()
        },
        **{
            name: value.reshape(batch_size, sequence_length)
            for name, value in rank_outputs.items()
        },
    }


def loss_and_hidden_gradient(
    model,
    states: torch.Tensor,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    aux_loss_weight: float,
    sequence_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Piano trainer loss, chunked over sequence positions for CUDA memory."""

    batch_size = states.shape[0]
    labels = input_ids[:, 1:]
    assistant = assistant_mask[:, 1:].float()
    valid = valid_mask[:, 1:].float()
    nonassistant = valid - assistant
    assistant_weights = (
        assistant / assistant.sum(-1, keepdim=True).clamp_min(1.0) / batch_size
    )
    nonassistant_weights = (
        nonassistant
        / nonassistant.sum(-1, keepdim=True).clamp_min(1.0)
        / batch_size
    )
    leaf = states.detach().requires_grad_(True)
    assistant_losses = []
    auxiliary_losses = []

    for start in range(0, states.shape[1], sequence_block_size):
        stop = min(states.shape[1], start + sequence_block_size)
        logits = model.lm_head(leaf[:, start:stop]).float()
        raw_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels[:, start:stop].reshape(-1),
            reduction="none",
        ).reshape(batch_size, stop - start)
        assistant_loss = (
            raw_loss * assistant_weights[:, start:stop]
        ).sum()
        auxiliary_loss = (
            raw_loss * nonassistant_weights[:, start:stop]
        ).sum()
        (assistant_loss + aux_loss_weight * auxiliary_loss).backward()
        assistant_losses.append(assistant_loss.detach())
        auxiliary_losses.append(auxiliary_loss.detach())

    return (
        torch.stack(assistant_losses).sum(),
        torch.stack(auxiliary_losses).sum(),
        leaf.grad.detach().to(states.dtype),
    )


def discover_subsets(dataset_name: str, requested) -> list[str]:
    available = sorted(datasets.get_dataset_config_names(dataset_name))
    if requested is None:
        return available
    selected = list(requested)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown subsets: {unknown}. Available subsets: {available}"
        )
    if not selected:
        raise ValueError("sampling.subsets must contain at least one subset")
    return selected


def sample_trajectories(config: DictConfig) -> tuple[list[Trajectory], list[str]]:
    subsets = discover_subsets(config.data.dataset, config.sampling.subsets)
    if config.sampling.latents is not None:
        requested_latents = [str(value) for value in config.sampling.latents]
        if not requested_latents:
            raise ValueError("sampling.latents must contain at least one latent")
        if len(set(requested_latents)) != len(requested_latents):
            raise ValueError("sampling.latents must not contain duplicates")

        requested_set = set(requested_latents)
        found: dict[str, Trajectory] = {}
        for subset in subsets:
            stream = datasets.load_dataset(
                config.data.dataset,
                subset,
                split="train",
                streaming=True,
            )
            iterator = iter(stream)
            for dataset_index, row in enumerate(iterator):
                latent = str(row.get("latent") or "")
                if latent not in requested_set or latent in found:
                    continue
                found[latent] = Trajectory(
                    uid=-1,
                    subset=subset,
                    source=str(row.get("source") or subset.replace("--", "/")),
                    latent=latent,
                    dataset_index=dataset_index,
                    row=row,
                )
                if len(found) == len(requested_latents):
                    break
            del iterator, stream
            gc.collect()
            if len(found) == len(requested_latents):
                break

        missing = [latent for latent in requested_latents if latent not in found]
        if missing:
            raise ValueError(
                f"Could not find requested latents in the selected subsets: {missing}"
            )
        trajectories = [found[latent] for latent in requested_latents]
        for uid, trajectory in enumerate(trajectories):
            trajectory.uid = uid
        return trajectories, subsets

    n = int(config.sampling.n)
    pool_size = int(config.sampling.trajectory_pool_size)
    if n < 1:
        raise ValueError("sampling.n must be at least 1")
    if pool_size < n:
        raise ValueError("sampling.trajectory_pool_size must be >= sampling.n")

    trajectories = []
    next_uid = 0
    for subset_index, subset in enumerate(subsets):
        stream = datasets.load_dataset(
            config.data.dataset,
            subset,
            split="train",
            streaming=True,
        )
        iterator = iter(stream)
        pool = []
        for _ in range(pool_size):
            try:
                pool.append(next(iterator))
            except StopIteration:
                break
        if len(pool) < n:
            raise ValueError(
                f"Subset {subset!r} contains only {len(pool)} trajectories, "
                f"fewer than sampling.n={n}"
            )
        rng = np.random.default_rng(int(config.seed) + subset_index * 1009)
        selected_indices = rng.choice(len(pool), n, replace=False)
        for dataset_index in selected_indices:
            row = pool[int(dataset_index)]
            trajectories.append(
                Trajectory(
                    uid=next_uid,
                    subset=subset,
                    source=str(row.get("source") or subset.replace("--", "/")),
                    latent=str(row.get("latent") or f"trajectory-{dataset_index}"),
                    dataset_index=int(dataset_index),
                    row=row,
                )
            )
            next_uid += 1
        del iterator, stream, pool
        gc.collect()

    if config.sampling.mode == "episodes":
        total = len(trajectories)
        episode_count = int(config.data.episode_count)
        positions = np.rint(
            (np.arange(total) + 0.5) * episode_count / total - 0.5
        ).astype(int)
        positions = np.clip(positions, 0, episode_count - 1)
        rng = np.random.default_rng(int(config.seed) + 9173)
        rng.shuffle(positions)
        for trajectory, position in zip(trajectories, positions):
            trajectory.target_episode = int(position)

    return trajectories, subsets


def load_model(config: DictConfig, device: torch.device):
    model_config = OmegaConf.create(
        OmegaConf.to_container(config.model, resolve=True)
    )
    model = import_model(model_config.type)(model_config)
    if model_config.pretrained_url is not None:
        load_checkpoint_state(
            model,
            model_config.pretrained_url,
            int(model_config.pretrained_step),
            strict=bool(model_config.pretrained_strict),
            verbose=True,
        )
    model = model.float().to(device).train()
    if config.runtime.compute_dtype != "float32":
        model.lm_head.to(getattr(torch, config.runtime.compute_dtype))
    enable_gradient_checkpointing(
        model, bool(config.runtime.gradient_checkpointing)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def write_episode_data(
    output_root: Path,
    trajectory: Trajectory,
    position: int,
    tokenizer,
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    gates: np.ndarray,
    statistics: dict[str, torch.Tensor],
    batch_index: int,
) -> dict[str, Any]:
    length = int(valid_mask[batch_index].sum())
    prediction_count = max(0, length - 1)
    input_token_ids = input_ids[
        batch_index, :prediction_count
    ].detach().cpu().numpy().astype(np.int64)
    target_token_ids = input_ids[
        batch_index, 1 : prediction_count + 1
    ].detach().cpu().numpy().astype(np.int64)
    target_assistant = assistant_mask[
        batch_index, 1 : prediction_count + 1
    ].detach().cpu().numpy().astype(bool)

    data_dir = output_root / "data" / f"trajectory_{trajectory.uid:04d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"episode_{position:02d}.npz"
    arrays = {
        "gates": gates,
        "input_ids": input_token_ids,
        "target_ids": target_token_ids,
        "target_assistant": target_assistant,
        "input_pieces": decoded_pieces(tokenizer, input_token_ids),
        "target_pieces": decoded_pieces(tokenizer, target_token_ids),
    }
    for name, values in statistics.items():
        arrays[name] = (
            values[batch_index, :prediction_count]
            .detach()
            .cpu()
            .numpy()
        )
    np.savez_compressed(data_path, **arrays)
    return {
        "trajectory_uid": trajectory.uid,
        "subset": trajectory.subset,
        "source": trajectory.source,
        "latent": trajectory.latent,
        "dataset_index": trajectory.dataset_index,
        "episode": position,
        "prediction_tokens": prediction_count,
        "assistant_targets": int(target_assistant.sum()),
        "nonassistant_targets": int((~target_assistant).sum()),
        "data_file": str(data_path.relative_to(output_root)),
    }


def collect(
    config: DictConfig,
    model,
    trajectories: list[Trajectory],
    output_root: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    collator = HorizonCollator(
        tokenizer_url=config.data.tokenizer_url,
        max_length=int(config.data.max_length),
        cluster_length=int(config.data.episode_count),
    )
    train_first = model_mode(model, "TRAIN_FIRST")
    inference = model_mode(model, "INFERENCE")
    entries = []
    batch_size = int(config.runtime.batch_size)

    for batch_start in range(0, len(trajectories), batch_size):
        selected = trajectories[batch_start : batch_start + batch_size]
        batch = collator([trajectory.row for trajectory in selected])
        input_ids, assistant_mask, valid_mask = (
            batch[key].to(device)
            for key in ("input_ids", "assistant_mask", "attention_mask")
        )
        actual_batch_size = len(selected)
        model.init_state(actual_batch_size, device)
        if config.sampling.mode == "episodes":
            record_positions = [
                {int(trajectory.target_episode)} for trajectory in selected
            ]
        else:
            record_positions = [
                set(range(int(config.data.episode_count))) for _ in selected
            ]
        collector = GateCollector(model, record_positions)
        maximum_position = max(max(positions) for positions in record_positions)

        try:
            for position in range(maximum_position + 1):
                collector.position = position
                ids = input_ids[:, position]
                assistant = assistant_mask[:, position]
                valid = valid_mask[:, position]
                should_record = any(
                    position in positions for positions in record_positions
                )

                empty_states = None
                if should_record:
                    with empty_fast_weights(model), torch.no_grad(), autocast_context(
                        device, config.runtime.compute_dtype
                    ):
                        empty_states, _ = model.forward(
                            input_ids=ids,
                            shift_states=True,
                            compute_logits=False,
                            mode=inference,
                            valid_mask=valid,
                        )

                with autocast_context(device, config.runtime.compute_dtype):
                    updated_states, _ = model.forward(
                        input_ids=ids,
                        shift_states=True,
                        compute_logits=False,
                        mode=train_first,
                        valid_mask=valid,
                    )

                if should_record:
                    with autocast_context(device, config.runtime.compute_dtype):
                        statistics = prediction_statistics(
                            model,
                            updated_states,
                            empty_states,
                            ids[:, 1:],
                            valid[:, 1:],
                            int(config.runtime.diagnostic_token_block_size),
                        )
                    for batch_index, trajectory in enumerate(selected):
                        if position not in record_positions[batch_index]:
                            continue
                        gates = collector.get(batch_index, position)
                        entries.append(
                            write_episode_data(
                                output_root,
                                trajectory,
                                position,
                                collator.tokenizer,
                                ids,
                                assistant,
                                valid,
                                gates,
                                statistics,
                                batch_index,
                            )
                        )
                    del statistics, empty_states

                with autocast_context(device, config.runtime.compute_dtype):
                    assistant_loss, auxiliary_loss, gradient = (
                        loss_and_hidden_gradient(
                            model,
                            updated_states,
                            ids,
                            assistant,
                            valid,
                            float(config.adaptation.aux_loss_weight),
                            int(config.runtime.loss_sequence_block_size),
                        )
                    )
                gradient_scale = (
                    actual_batch_size
                    / float(config.adaptation.reference_batch_size)
                )
                torch.autograd.backward(
                    updated_states,
                    gradient * gradient_scale,
                    inputs=model.grad_containers(),
                )
                model.update_state(train_first)
                print(
                    f"batch={batch_start // batch_size + 1} "
                    f"episode={position:02d} "
                    f"assistant_loss={assistant_loss.item():.5f} "
                    f"aux_loss={auxiliary_loss.item():.5f}",
                    flush=True,
                )
                del updated_states, gradient
        finally:
            collector.remove()

        del input_ids, assistant_mask, valid_mask, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return entries


def quantile(values: np.ndarray, q: float, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if len(finite) else default


def load_html_example(output_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    logged = np.load(output_root / entry["data_file"])
    gates = logged["gates"].astype(np.float64)
    means = np.maximum(gates.mean(axis=1), 1e-30)
    p95 = np.maximum(np.quantile(gates, 0.95, axis=1), 1e-30)
    logp_updated = logged["logp_updated"].astype(np.float64)
    logp_empty = logged["logp_empty"].astype(np.float64)
    delta = logp_updated - logp_empty
    kl = logged["kl"].astype(np.float64)
    return {
        **{key: entry[key] for key in (
            "trajectory_uid", "subset", "source", "latent",
            "dataset_index", "episode",
        )},
        "inputTokens": [token_label(piece) for piece in logged["input_pieces"]],
        "targetTokens": [token_label(piece) for piece in logged["target_pieces"]],
        "fullTextPieces": logged["target_pieces"].astype(str).tolist(),
        "assistant": logged["target_assistant"].astype(bool).tolist(),
        "normalizedGates": (gates / means[:, None]).tolist(),
        "gateColors": (gates / p95[:, None]).tolist(),
        "kl": kl.tolist(),
        "logpUpdated": logp_updated.tolist(),
        "logpEmpty": logp_empty.tolist(),
        "deltaLogp": delta.tolist(),
        "rankUpdated": logged["rank_updated"].astype(int).tolist(),
        "rankEmpty": logged["rank_empty"].astype(int).tolist(),
        "scales": {
            "kl": max(quantile(kl, 0.95, 1.0), 1e-12),
            "logpLow": quantile(logp_updated, 0.05, -10.0),
            "logpHigh": quantile(logp_updated, 0.95, 0.0),
            "logpBaseLow": quantile(logp_empty, 0.05, -10.0),
            "logpBaseHigh": quantile(logp_empty, 0.95, 0.0),
            "delta": max(quantile(np.abs(delta), 0.95, 1.0), 1e-12),
        },
    }


def build_viewer(
    output_root: Path,
    entries: list[dict[str, Any]],
    output_path: Path,
    note: str | None,
) -> None:
    examples = [load_html_example(output_root, entry) for entry in entries]
    examples.sort(
        key=lambda example: (
            example["source"].casefold(),
            example["latent"].casefold(),
            example["dataset_index"],
            example["episode"],
        )
    )
    payload = json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")
    note_payload = json.dumps(note or "", ensure_ascii=False).replace("</", "<\\/")
    document = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Per-token Adaptive Learning Rate Scales</title>
<style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,sans-serif}body{margin:0;background:#f6f7fb;color:#172033}header{padding:20px 24px 15px;background:white;border-bottom:1px solid #dfe3eb}h1{margin:0 0 5px;font-size:24px}.controls{display:flex;align-items:center;gap:10px;margin:10px 0 7px}.controls label{color:#465268;font-size:13px;font-weight:600}select{min-width:720px;max-width:94vw;padding:7px 30px 7px 9px;border:1px solid #b9c1ce;border-radius:6px;background:white;color:#172033;font:13px ui-monospace,monospace}.subtitle{color:#667085;font:12px ui-monospace,monospace;margin-bottom:13px}.preamble{max-width:1100px;color:#465268;font-size:14px;line-height:1.48}.preamble p{margin:7px 0}.note{display:none;max-width:1100px;margin:12px 0 0;padding:10px 12px;border-left:4px solid #7863b6;background:#f4f0ff;white-space:pre-wrap;font-size:13px}.roles{display:flex;gap:14px;margin-top:11px;font-size:12px;color:#667085}.role:before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.role.a:before{background:#e28a20}.role.n:before{background:#b13aa3}.fulltext{margin-top:13px;padding:11px 13px;max-height:150px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;border:1px solid #d8dde7;border-radius:7px;font:13px/1.48 ui-monospace,monospace}.fulltext span{cursor:pointer;border-radius:2px;transition:background-color .08s,box-shadow .08s;box-decoration-break:clone;-webkit-box-decoration-break:clone}.fulltext .a{--role-color:#e28a20;background:#fff0d8}.fulltext .n{--role-color:#b13aa3;background:#f8e8f5}.fulltext.heatmap-active span{background:var(--token-heat)!important;color:var(--token-ink);box-shadow:inset 0 0 0 2px var(--role-color)}.fulltext .visible-token{outline:1px solid #8b5cf6;outline-offset:-1px}.fulltext .linked-hover{outline:2px solid #d97706;outline-offset:-2px}.fulltext .linked-selected{outline:2px solid #2563eb;outline-offset:-2px}.viewport{overflow:auto;padding:16px 18px 24px;min-height:400px}.matrix{display:flex;width:max-content;align-items:flex-start;background:white;border:1px solid #d8dde7;box-shadow:0 4px 18px #29344d14}.labels{width:116px;flex:none;position:sticky;left:0;z-index:5;background:#f8fafc;border-right:2px solid #aeb7c7}.label-top,.label-bottom{height:76px;box-sizing:border-box;display:flex;align-items:center;justify-content:center;text-align:center;padding:6px;font-size:11px;color:#667085}.row-label{height:24px;display:flex;align-items:center;justify-content:center;border-top:1px solid #e5e8ee;font:10px ui-monospace,monospace;text-align:center;padding:0 4px;cursor:pointer;user-select:none}.row-label:hover,.row-label:focus-visible{background:#ede9fe;outline:2px solid #8b5cf6;outline-offset:-2px}.row-label.context-selected{background:#ddd6fe;color:#4c1d95;font-weight:700;box-shadow:inset 3px 0 #7c3aed}.row-label.diag{height:28px;font-weight:600}.label-bottom{border-top:2px solid #aeb7c7}.token-col{width:72px;flex:none;border-right:1px solid #e6e9ef}.token-box{height:76px;box-sizing:border-box;padding:5px 3px;display:flex;flex-direction:column;align-items:center;justify-content:center;overflow:hidden;background:#fbfcfe;cursor:pointer}.target{border-top:4px solid #b13aa3}.target.a{border-top-color:#e28a20;background:#fffbf5}.input{border-top:2px solid #aeb7c7;background:#f8fafc}.token-box:hover,.token-box.linked-hover{background:#fef3c7;box-shadow:inset 0 0 0 2px #d97706}.token-box.linked-selected{background:#dbeafe;box-shadow:inset 0 0 0 2px #2563eb}.token{white-space:normal;overflow:hidden;overflow-wrap:anywhere;max-height:50px;line-height:1.15;text-align:center;font:12px ui-monospace,monospace}.index{font:9px ui-monospace,monospace;color:#7b8494;margin-top:3px}.cell{height:24px;box-sizing:border-box;border-top:1px solid #ffffff73;cursor:crosshair}.cell.diag{height:28px}.cell:hover{outline:2px solid #111827;outline-offset:-2px}.tip{position:fixed;pointer-events:none;display:none;z-index:20;background:#111827;color:white;padding:7px 9px;border-radius:6px;white-space:pre;font:12px/1.45 ui-monospace,monospace;box-shadow:0 4px 14px #0005}
</style></head><body><header>
<h1>Per-token Adaptive Learning Rate Scales</h1>
<div class="controls"><label for="example">Example</label><select id="example"></select></div>
<div class="subtitle" id="subtitle"></div>
<div class="preamble">
<p>This visualization shows the adaptive learning rate scales of the fast weight updates at each layer and token.</p>
<p>The columns are aligned such that the corresponding hidden states take in the bottom token and predict the top token.</p>
<p>The gate color gradient is linear, ranging from 0 (white) to each layer's 95th percentile (red), with larger values clipped. KL uses white-to-dark-green; p updated and p base map probability directly from 0 (white) to 1 (blue); Δ log p vs base uses blue-to-white-to-dark-green.</p>
<p>Hovering a gate shows its value normalized by its layer's mean. Hovering a prediction cell shows the updated and empty-fast-weight log likelihoods and label ranks.</p>
<p>Click or hover a token to locate its counterpart in the context or table. Tokens currently visible in the table are boxed in the context.</p>
<p>Click a prediction or layer label in the left legend to apply that row's color gradient to the context. Orange and pinkish-purple token perimeters continue to indicate assistant and non-assistant targets.</p>
</div><div class="note" id="note"></div>
<div class="roles"><span class="role a">assistant target</span><span class="role n">non-assistant target</span></div>
<div class="fulltext" id="fulltext"></div></header>
<div class="viewport" id="viewport"><div class="matrix" id="matrix"></div></div><div class="tip" id="tip"></div>
<script>
const EXAMPLES=__PAYLOAD__, NOTE=__NOTE__;
const picker=document.getElementById('example'),matrix=document.getElementById('matrix'),tip=document.getElementById('tip'),fulltext=document.getElementById('fulltext'),subtitle=document.getElementById('subtitle'),viewport=document.getElementById('viewport'),note=document.getElementById('note');
if(NOTE){note.textContent=NOTE;note.style.display='block'}
const nameWidth=Math.max(...EXAMPLES.map(d=>d.latent.length)),horizonWidth=Math.max(...EXAMPLES.map(d=>String(d.dataset_index).length)),episodeWidth=Math.max(...EXAMPLES.map(d=>String(d.episode).length));
const fixed=s=>s.replaceAll(' ','\u00a0'),groups=new Map();
EXAMPLES.forEach((d,i)=>{if(!groups.has(d.source)){const g=document.createElement('optgroup');g.label=d.source;groups.set(d.source,g);picker.appendChild(g)}const o=document.createElement('option');o.value=i;o.textContent=fixed(`${d.latent.padEnd(nameWidth)}  ·  trajectory ${String(d.dataset_index).padStart(horizonWidth)}  ·  episode ${String(d.episode).padStart(episodeWidth)}`);groups.get(d.source).appendChild(o)});
function blend(lo,hi,t){t=Math.max(0,Math.min(1,t));return `rgb(${lo.map((x,i)=>Math.round(x+(hi[i]-x)*t)).join(',')})`}
const gateColor=v=>blend([255,255,255],[160,24,37],v);
const positiveRowColor=t=>blend([255,255,255],[20,105,75],t);
const probabilityColor=p=>blend([255,255,255],[49,130,189],p);
function topRowColor(t){t=Math.max(0,Math.min(1,t));return t<.5?blend([49,130,189],[255,255,255],t*2):blend([255,255,255],[20,105,75],(t-.5)*2)}
const klColor=v=>positiveRowColor(v);
function deltaColor(v,scale){return topRowColor(.5+.5*Math.max(-1,Math.min(1,v/scale)))}
function readableInk(color){const rgb=(color.match(/\d+/g)||[255,255,255]).map(Number),luma=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];return luma<145?'#fff':'#172033'}
function diagnosticTip(D,j){return `KL(updated ‖ base): ${D.kl[j].toFixed(6)}\nΔ log p vs base:   ${D.deltaLogp[j].toFixed(4)}\np updated:         ${Math.exp(D.logpUpdated[j]).toPrecision(6)}\np base:            ${Math.exp(D.logpEmpty[j]).toPrecision(6)}\nlog p updated:     ${D.logpUpdated[j].toFixed(4)}\nlog p base (empty): ${D.logpEmpty[j].toFixed(4)}\nrank updated:      ${D.rankUpdated[j].toLocaleString()}\nrank base (empty): ${D.rankEmpty[j].toLocaleString()}`}
function attachTip(cell,text){cell.onmouseenter=()=>{tip.style.display='block';tip.textContent=text};cell.onmousemove=e=>{tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'};cell.onmouseleave=()=>tip.style.display='none'}
let selectedToken=-1,selectedContextRow=-1,visibleFrame=0;
const contextTokens=()=>Array.from(fulltext.querySelectorAll('[data-token-index]')),tableColumns=()=>Array.from(matrix.querySelectorAll('.token-col')),tableTarget=index=>tableColumns()[index]?.querySelector('.target');
function mirrorHover(index,on){const span=contextTokens()[index],target=tableTarget(index);if(span)span.classList.toggle('linked-hover',on);if(target)target.classList.toggle('linked-hover',on)}
function scrollTableTo(index){const col=tableColumns()[index];if(!col)return;viewport.scrollTo({left:Math.max(0,col.offsetLeft-viewport.clientWidth/2+col.offsetWidth/2),behavior:'smooth'})}
function scrollContextTo(index){const span=contextTokens()[index];if(!span)return;fulltext.scrollTo({top:Math.max(0,span.offsetTop-fulltext.clientHeight/2+span.offsetHeight/2),behavior:'smooth'})}
function selectLinked(index,origin){const spans=contextTokens();if(selectedToken>=0){spans[selectedToken]?.classList.remove('linked-selected');tableTarget(selectedToken)?.classList.remove('linked-selected')}selectedToken=index;spans[index]?.classList.add('linked-selected');tableTarget(index)?.classList.add('linked-selected');if(origin==='context')scrollTableTo(index);else scrollContextTo(index)}
function linkToken(element,index,origin){element.addEventListener('mouseenter',()=>mirrorHover(index,true));element.addEventListener('mouseleave',()=>mirrorHover(index,false));element.addEventListener('click',()=>selectLinked(index,origin))}
function contextRowColor(D,row,index){if(row===0)return klColor(D.kl[index]/D.scales.kl);if(row===1)return probabilityColor(Math.exp(D.logpUpdated[index]));if(row===2)return probabilityColor(Math.exp(D.logpEmpty[index]));if(row===3)return deltaColor(D.deltaLogp[index],D.scales.delta);return gateColor(D.gateColors[row-4][index])}
function selectContextRow(D,labels,row){const rowLabels=Array.from(labels.querySelectorAll('.row-label'));if(selectedContextRow===row){selectedContextRow=-1;fulltext.classList.remove('heatmap-active');rowLabels[row]?.classList.remove('context-selected');contextTokens().forEach(span=>{span.style.removeProperty('--token-heat');span.style.removeProperty('--token-ink')});return}if(selectedContextRow>=0)rowLabels[selectedContextRow]?.classList.remove('context-selected');selectedContextRow=row;rowLabels[row]?.classList.add('context-selected');fulltext.classList.add('heatmap-active');contextTokens().forEach((span,index)=>{const color=contextRowColor(D,row,index);span.style.setProperty('--token-heat',color);span.style.setProperty('--token-ink',readableInk(color))})}
function activateRowLabels(D,labels){Array.from(labels.querySelectorAll('.row-label')).forEach((label,row)=>{label.tabIndex=0;label.setAttribute('role','button');label.title='Apply this row color scale to the context';label.addEventListener('click',()=>selectContextRow(D,labels,row));label.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectContextRow(D,labels,row)}})})}
function updateVisibleTokens(){visibleFrame=0;const spans=contextTokens(),cols=tableColumns(),labels=matrix.querySelector('.labels'),viewportRect=viewport.getBoundingClientRect(),left=Math.max(viewportRect.left,labels?labels.getBoundingClientRect().right:viewportRect.left),right=viewportRect.right;spans.forEach(span=>span.classList.remove('visible-token'));cols.forEach((col,index)=>{const rect=col.getBoundingClientRect();if(rect.right>left&&rect.left<right)spans[index]?.classList.add('visible-token')})}
function scheduleVisibleUpdate(){if(!visibleFrame)visibleFrame=requestAnimationFrame(updateVisibleTokens)}
function render(index){const D=EXAMPLES[index];selectedToken=-1;selectedContextRow=-1;tip.style.display='none';matrix.replaceChildren();fulltext.replaceChildren();fulltext.classList.remove('heatmap-active');fulltext.scrollTop=0;viewport.scrollLeft=0;subtitle.textContent=`Episode ${D.episode} · Source: ${D.source} · Repository: ${D.latent} · Trajectory ${D.dataset_index}`;D.fullTextPieces.forEach((piece,j)=>{const s=document.createElement('span');s.className=D.assistant[j]?'a':'n';s.dataset.tokenIndex=j;s.textContent=piece;linkToken(s,j,'context');attachTip(s,diagnosticTip(D,j));fulltext.appendChild(s)});const labels=document.createElement('div');labels.className='labels';labels.innerHTML='<div class="label-top">token predicted<br>at top</div><div class="row-label diag">KL(updated ‖ base)</div><div class="row-label diag">p updated</div><div class="row-label diag">p base</div><div class="row-label diag">Δ log p vs base</div>'+Array.from({length:D.normalizedGates.length},(_,l)=>`<div class="row-label">layer ${l}</div>`).join('')+'<div class="label-bottom">token taken in<br>at bottom</div>';matrix.appendChild(labels);activateRowLabels(D,labels);D.targetTokens.forEach((target,j)=>{const col=document.createElement('div');col.className='token-col';col.dataset.tokenIndex=j;const top=document.createElement('div');top.className='token-box target'+(D.assistant[j]?' a':'');top.innerHTML='<div class="token"></div><div class="index">'+(j+1)+'</div>';top.querySelector('.token').textContent=target;linkToken(top,j,'table');col.appendChild(top);const colors=[klColor(D.kl[j]/D.scales.kl),probabilityColor(Math.exp(D.logpUpdated[j])),probabilityColor(Math.exp(D.logpEmpty[j])),deltaColor(D.deltaLogp[j],D.scales.delta)];colors.forEach(c=>{const cell=document.createElement('div');cell.className='cell diag';cell.style.background=c;attachTip(cell,diagnosticTip(D,j));col.appendChild(cell)});for(let l=0;l<D.normalizedGates.length;l++){const cell=document.createElement('div');cell.className='cell';cell.style.background=gateColor(D.gateColors[l][j]);attachTip(cell,D.normalizedGates[l][j].toFixed(3));col.appendChild(cell)}const bottom=document.createElement('div');bottom.className='token-box input';bottom.innerHTML='<div class="token"></div><div class="index">'+j+'</div>';bottom.querySelector('.token').textContent=D.inputTokens[j];if(j>0)linkToken(bottom,j-1,'table');else bottom.style.cursor='default';col.appendChild(bottom);matrix.appendChild(col)});scheduleVisibleUpdate()}
viewport.addEventListener('scroll',scheduleVisibleUpdate,{passive:true});window.addEventListener('resize',scheduleVisibleUpdate,{passive:true});picker.addEventListener('change',()=>render(Number(picker.value)));render(0);
</script></body></html>'''.replace("__PAYLOAD__", payload).replace("__NOTE__", note_payload)
    output_path.write_text(document)


def build_index(
    output_path: Path,
    files: list[tuple[dict[str, Any], Path]],
    note: str | None,
) -> None:
    grouped: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for entry, path in files:
        grouped.setdefault(entry["source"], []).append((entry, path))
    sections = []
    for source in sorted(grouped, key=str.casefold):
        links = []
        for entry, path in sorted(
            grouped[source], key=lambda item: (
                item[0]["latent"].casefold(), item[0]["dataset_index"]
            )
        ):
            links.append(
                f'<li><a href="{html.escape(path.name)}">'
                f'{html.escape(entry["latent"])} · trajectory '
                f'{entry["dataset_index"]}</a></li>'
            )
        sections.append(
            f"<h2>{html.escape(source)}</h2><ul>{''.join(links)}</ul>"
        )
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    output_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Token gate trajectories</title>"
        "<style>body{max-width:1000px;margin:32px auto;padding:0 20px;font-family:system-ui;color:#172033}"
        "h2{margin-top:28px}li{margin:8px 0}a{color:#315f9f}.note{white-space:pre-wrap;padding:12px;background:#f4f0ff;border-left:4px solid #7863b6}</style>"
        "</head><body><h1>Per-token Adaptive Learning Rate Scales</h1>"
        + note_html
        + "".join(sections)
        + "</body></html>"
    )


def write_outputs(
    config: DictConfig,
    data_root: Path,
    run_output_root: Path,
    entries: list[dict[str, Any]],
) -> list[Path]:
    note = None if config.output.note is None else str(config.output.note)
    outputs = []
    if config.sampling.mode == "episodes":
        output_path = run_output_root / "index.html"
        build_viewer(data_root, entries, output_path, note)
        outputs.append(output_path)
    else:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for entry in entries:
            grouped.setdefault(entry["trajectory_uid"], []).append(entry)
        index_files = []
        for uid, trajectory_entries in sorted(grouped.items()):
            first = trajectory_entries[0]
            filename = (
                f"{slug(first['source'])}__{slug(first['latent'])}__"
                f"trajectory_{first['dataset_index']:05d}.html"
            )
            output_path = run_output_root / filename
            build_viewer(data_root, trajectory_entries, output_path, note)
            outputs.append(output_path)
            index_files.append((first, output_path))
        index_path = run_output_root / "index.html"
        build_index(index_path, index_files, note)
        outputs.insert(0, index_path)
    return outputs


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="visualize_piano_token_gates",
)
def main(config: DictConfig) -> None:
    if config.sampling.latents is not None:
        config.sampling.mode = "trajectories"
    if config.sampling.mode not in ("episodes", "trajectories"):
        raise ValueError("sampling.mode must be 'episodes' or 'trajectories'")
    if int(config.data.episode_count) < 1:
        raise ValueError("data.episode_count must be at least 1")
    device = torch.device(str(config.runtime.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda but CUDA is unavailable")
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.set_float32_matmul_precision("high")

    configured_output = Path(str(config.output.directory))
    output_base = (
        configured_output
        if configured_output.is_absolute()
        else constants.BASE_PATH / configured_output
    )
    model_output_root = output_base / model_url_folder(config.model)
    name = filename_stem(config.name)
    run_output_root = model_output_root / name
    run_output_root.mkdir(parents=True, exist_ok=True)
    data_root = run_output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(config), flush=True)
    trajectories, subsets = sample_trajectories(config)
    model = load_model(config, device)
    entries = collect(config, model, trajectories, data_root, device)
    outputs = write_outputs(
        config, data_root, run_output_root, entries
    )

    manifest = {
        "config": OmegaConf.to_container(config, resolve=True),
        "subsets": subsets,
        "trajectory_count": len(trajectories),
        "visualized_episode_count": len(entries),
        "data_directory": str(data_root.relative_to(run_output_root)),
        "state_update_note": (
            "Each visualized episode uses fast weights updated through every "
            "preceding episode in its trajectory. Empty probabilities use zero "
            "fast weights. Adaptation gradients include the configured auxiliary loss."
        ),
        "entries": entries,
        "html_files": [
            str(path.relative_to(run_output_root)) for path in outputs
        ],
    }
    manifest_path = run_output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Generated:", flush=True)
    for path in outputs:
        print(path, flush=True)


if __name__ == "__main__":
    main()
