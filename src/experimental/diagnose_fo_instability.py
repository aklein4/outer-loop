from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace

import datasets
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from collators.horizon import HorizonCollator
from models.fo_ittt import (
    FastWeightMLP,
    FoItttModel,
    _raw_fast_weight_gradient,
)
import models.fo_ittt as fo_ittt
from utils.sharding_utils import maybe_shard_with_gradients
from utils.torch_utils import newton_schulz


PROBE = SimpleNamespace(
    phase="idle",
    episode=-1,
    inject_activation=True,
    weight_to_layer={},
    backward=[],
    updates=[],
    forwards=[],
    trace_layers=(),
    raw_tensors=[],
    update_tensors=[],
    use_reference_replay=False,
    reference_raw={},
    use_reference_local_jacobian=False,
    reference_local={},
    meta_grad_eps=None,
    detach_meta_rms=False,
    future_gradient_mode="full",
)


def scalar_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
    x = x.detach().float()
    return {
        "rms": x.square().mean().sqrt(),
        "norm": x.norm(),
        "max": x.abs().max(),
    }


def record(kind: str, layer: int, **values):
    target = getattr(PROBE, kind)
    target.append(
        {
            "phase": PROBE.phase,
            "episode": PROBE.episode,
            "layer": layer,
            **{key: value.detach() for key, value in values.items()},
        }
    )


class ProbeFastWeightFunction(torch.autograd.Function):
    """The training function with scalar diagnostics added."""

    @staticmethod
    def forward(
        ctx,
        activations,
        output,
        down_weight,
        grad_buffer,
        remaining_gradient,
        learning_rate,
        grad_eps,
    ):
        batch_size = grad_buffer.shape[0]
        ctx.second_pass = remaining_gradient is not None
        expected_batch_size = batch_size * (2 if ctx.second_pass else 1)
        if activations.shape[0] != expected_batch_size:
            raise ValueError(
                f"expected fast-weight batch {expected_batch_size}, "
                f"got {activations.shape[0]}"
            )

        ctx.layer = PROBE.weight_to_layer[down_weight.data_ptr()]
        if ctx.second_pass:
            ctx.save_for_backward(
                activations.reshape(
                    batch_size,
                    2,
                    *activations.shape[1:],
                )[:, 0],
                down_weight,
                remaining_gradient,
                learning_rate,
            )
            ctx.activation_dtype = activations.dtype
        else:
            ctx.save_for_backward(activations, down_weight)
        ctx.grad_dtype = grad_buffer.dtype
        ctx.grad_eps = grad_eps
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        layer = ctx.layer
        if not ctx.second_pass:
            activations, down_weight = ctx.saved_tensors
            raw_gradient = _raw_fast_weight_gradient(
                activations,
                output_gradient,
                down_weight,
                ctx.grad_dtype,
            )
            if PROBE.use_reference_replay:
                key = (PROBE.episode, layer)
                if PROBE.phase == "first":
                    PROBE.reference_raw[key] = (
                        raw_gradient.detach().clone()
                    )
                    if PROBE.use_reference_local_jacobian:
                        PROBE.reference_local[key] = (
                            activations.detach().clone(),
                            output_gradient.detach().clone(),
                        )
                elif PROBE.phase == "terminal":
                    raw_gradient = PROBE.reference_raw[key]
            raw = scalar_stats(raw_gradient)
            record(
                "backward",
                layer,
                raw_rms=raw["rms"],
                raw_norm=raw["norm"],
                output_grad_rms=scalar_stats(output_gradient)["rms"],
            )
            if layer in PROBE.trace_layers:
                PROBE.raw_tensors.append(
                    {
                        "phase": PROBE.phase,
                        "episode": PROBE.episode,
                        "layer": layer,
                        "raw": raw_gradient.detach().float().cpu().clone(),
                    }
                )
            return (
                None,
                output_gradient,
                None,
                raw_gradient,
                None,
                None,
                None,
            )

        (
            activations,
            down_weight,
            remaining_gradient,
            learning_rate,
        ) = ctx.saved_tensors
        batch_size = remaining_gradient.shape[0]
        lm_output_gradient = output_gradient.reshape(
            batch_size,
            2,
            *output_gradient.shape[1:],
        )[:, 0].detach()
        if PROBE.use_reference_local_jacobian:
            (
                local_activations,
                lm_output_gradient,
            ) = PROBE.reference_local[
                (PROBE.episode, layer)
            ]
        else:
            local_activations = activations

        with torch.enable_grad():
            activations_for_grad = (
                local_activations.detach().requires_grad_(True)
            )
            down_weight_for_grad = down_weight.detach().requires_grad_(True)
            learning_rate_for_grad = (
                learning_rate.detach().float().requires_grad_(True)
            )
            local_raw_gradient = _raw_fast_weight_gradient(
                activations_for_grad,
                lm_output_gradient,
                down_weight_for_grad,
                ctx.grad_dtype,
            )
            if PROBE.use_reference_replay:
                reference_raw = PROBE.reference_raw[
                    (PROBE.episode, layer)
                ].to(
                    device=local_raw_gradient.device,
                    dtype=local_raw_gradient.dtype,
                )
                # Use the first-pass value while retaining the replay
                # computation's local derivative with respect to parameters.
                local_raw_gradient = (
                    reference_raw
                    + local_raw_gradient
                    - local_raw_gradient.detach()
                )
                raw_gradient = reference_raw.detach()
            else:
                raw_gradient = local_raw_gradient.detach()
            future_gradient = (
                remaining_gradient.to(ctx.grad_dtype) - raw_gradient
            ).detach()
            effective_future_gradient = (
                torch.zeros_like(future_gradient)
                if PROBE.future_gradient_mode == "zero"
                else future_gradient
            )
            meta_grad_eps = (
                ctx.grad_eps
                if PROBE.meta_grad_eps is None
                else PROBE.meta_grad_eps
            )
            if PROBE.detach_meta_rms:
                # Preserve the forward value of the normalized update while
                # isolating the derivative through its RMS denominator.
                inverse_rms = torch.rsqrt(
                    local_raw_gradient.float().square().mean(
                        dim=(-2, -1),
                        keepdim=True,
                    )
                    + meta_grad_eps
                ).detach()
                normalized_gradient = (
                    local_raw_gradient.float() * inverse_rms
                ).to(local_raw_gradient.dtype)
            else:
                normalized_gradient = F.rms_norm(
                    local_raw_gradient,
                    local_raw_gradient.shape[-2:],
                    eps=meta_grad_eps,
                )
            state_update = -(
                learning_rate_for_grad * normalized_gradient
            )
            local_loss = (
                effective_future_gradient * state_update
            ).sum()
            (
                activation_gradient,
                down_weight_gradient,
                learning_rate_gradient,
            ) = torch.autograd.grad(
                local_loss,
                (
                    activations_for_grad,
                    down_weight_for_grad,
                    learning_rate_for_grad,
                ),
            )
            if PROBE.future_gradient_mode == "detach_lr":
                learning_rate_gradient = torch.zeros_like(
                    learning_rate_gradient
                )
            elif PROBE.future_gradient_mode == "detach_fast":
                activation_gradient = torch.zeros_like(
                    activation_gradient
                )
                down_weight_gradient = torch.zeros_like(
                    down_weight_gradient
                )

        raw_rms = raw_gradient.float().square().mean().sqrt()
        future_rms = future_gradient.float().square().mean().sqrt()
        remaining_rms = (
            remaining_gradient.float().square().mean().sqrt()
        )
        raw_flat = raw_gradient.float().flatten()
        future_flat = future_gradient.float().flatten()
        cosine = (
            torch.dot(raw_flat, future_flat)
            / (raw_flat.norm() * future_flat.norm()).clamp_min(1e-30)
        )
        record(
            "backward",
            layer,
            raw_rms=raw_rms,
            future_rms=future_rms,
            remaining_rms=remaining_rms,
            future_to_raw=future_rms / raw_rms.clamp_min(1e-30),
            raw_future_cosine=cosine,
            inverse_rms=torch.rsqrt(
                raw_gradient.float().square().mean() + ctx.grad_eps
            ),
            lr_rms=learning_rate.float().square().mean().sqrt(),
            lr_min=learning_rate.float().min(),
            lr_max=learning_rate.float().max(),
            state_update_rms=state_update.float().square().mean().sqrt(),
            injected_activation_grad_norm=activation_gradient.float().norm(),
            injected_down_grad_norm=down_weight_gradient.float().norm(),
            lr_tensor_grad_norm=learning_rate_gradient.float().norm(),
            example_raw_rms_min=raw_gradient.float().square().mean(
                dim=(-2, -1)
            ).sqrt().min(),
            example_raw_rms_max=raw_gradient.float().square().mean(
                dim=(-2, -1)
            ).sqrt().max(),
        )
        if layer in PROBE.trace_layers:
            PROBE.raw_tensors.append(
                {
                    "phase": PROBE.phase,
                    "episode": PROBE.episode,
                    "layer": layer,
                    "raw": raw_gradient.detach().float().cpu().clone(),
                    "future": future_gradient.detach().float().cpu().clone(),
                }
            )

        activation_gradient = torch.stack(
            (
                torch.zeros_like(activation_gradient),
                (
                    activation_gradient
                    if PROBE.inject_activation
                    else torch.zeros_like(activation_gradient)
                ),
            ),
            dim=1,
        ).flatten(0, 1).to(ctx.activation_dtype)
        activation_gradient = maybe_shard_with_gradients(
            activation_gradient
        )
        return (
            activation_gradient,
            output_gradient,
            down_weight_gradient.to(down_weight.dtype),
            raw_gradient,
            None,
            learning_rate_gradient.to(learning_rate.dtype),
            None,
        )


ORIGINAL_UPDATE_STATE = FastWeightMLP.update_state


@torch.no_grad()
def probe_update_state(
    self,
    state,
    grad_buffer,
    raw_gradient,
    embeddings,
    embedding_mask,
    subtract_gradients=False,
):
    normalized = F.rms_norm(
        raw_gradient.float(),
        raw_gradient.shape[-2:],
        eps=self.grad_eps,
    )
    learning_rate = self.get_lr(embeddings, embedding_mask)
    update = -learning_rate.float() * normalized
    layer = int(self.fast_weight_index.item())
    if layer in PROBE.trace_layers:
        PROBE.update_tensors.append(
            {
                "phase": PROBE.phase,
                "episode": PROBE.episode,
                "layer": layer,
                "state_before": state.detach().float().cpu().clone(),
                "raw": raw_gradient.detach().float().cpu().clone(),
                "learning_rate": (
                    learning_rate.detach().float().cpu().clone()
                ),
                "update": update.detach().float().cpu().clone(),
            }
        )
    record(
        "updates",
        layer,
        raw_rms=raw_gradient.float().square().mean().sqrt(),
        normalized_rms=normalized.square().mean().sqrt(),
        lr_rms=learning_rate.float().square().mean().sqrt(),
        lr_min=learning_rate.float().min(),
        lr_max=learning_rate.float().max(),
        update_rms=update.square().mean().sqrt(),
        state_rms_before=state.float().square().mean().sqrt(),
        embedding_rms=embeddings.float().square().mean().sqrt(),
    )
    return ORIGINAL_UPDATE_STATE(
        self,
        state,
        grad_buffer,
        raw_gradient,
        embeddings,
        embedding_mask,
        subtract_gradients=subtract_gradients,
    )


def install_probes(model: FoItttModel):
    fo_ittt.FastWeightFunction = ProbeFastWeightFunction
    FastWeightMLP.update_state = probe_update_state
    PROBE.weight_to_layer = {
        module.down_fast.weight.data_ptr(): index
        for index, module in enumerate(model._fast_weight_mlps())
    }

    def hook(index):
        def capture(module, args, output):
            if PROBE.phase != "first" or PROBE.episode not in (0, 7, 15):
                return
            x = args[0].detach().float()
            y = output.detach().float()
            state = model.fast_weight_state[index].detach().float()
            record(
                "forwards",
                index,
                input_rms=x.square().mean().sqrt(),
                output_rms=y.square().mean().sqrt(),
                state_rms=state.square().mean().sqrt(),
            )
        return capture

    handles = [
        module.register_forward_hook(hook(index))
        for index, module in enumerate(model._fast_weight_mlps())
    ]
    return handles


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def logit_loss_and_gradient(
    model,
    lm_states,
    labels,
    token_weights,
    num_iterations,
):
    lm_states = lm_states.detach().reshape(
        -1,
        num_iterations,
        lm_states.shape[-1],
    ).requires_grad_(True)
    labels = labels.reshape(-1, num_iterations)
    token_weights = token_weights.reshape(-1, num_iterations)
    losses = []
    for index in range(num_iterations):
        with autocast():
            logits = model.lm_head(
                lm_states[:, index].contiguous()
            ).float()
            loss = (
                F.cross_entropy(
                    logits,
                    labels[:, index],
                    reduction="none",
                )
                * token_weights[:, index]
            ).sum()
        losses.append(loss.detach())
        loss.backward()
    return torch.stack(losses).sum(), lm_states.grad.detach()


def loss_and_hidden_gradient(
    model,
    input_ids,
    assistant_mask,
    hidden_states,
    num_iterations,
):
    batch_size = hidden_states.shape[0]
    output_mask = assistant_mask[:, 1:].float()
    token_weights = (
        output_mask
        / output_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        / batch_size
    )
    hidden_leaf = hidden_states.detach().requires_grad_(True)
    with autocast():
        lm_states = model.model.norm(hidden_leaf)
    loss, lm_gradient = logit_loss_and_gradient(
        model,
        lm_states,
        input_ids[:, 1:],
        token_weights,
        num_iterations,
    )
    lm_states.backward(lm_gradient.reshape_as(lm_states))
    return loss, hidden_leaf.grad.detach().reshape_as(hidden_states)


def first_pass(
    model,
    input_ids,
    assistant_mask,
    attention_mask,
    num_iterations,
    update_state=True,
    fast_weight_gradients_only=True,
):
    model.set_fast_weight_mode(FastWeightMLP.FIRST_PASS)
    with autocast():
        hidden_states = model.backbone_forward(input_ids=input_ids)
        loss, hidden_gradient = loss_and_hidden_gradient(
            model,
            input_ids,
            assistant_mask,
            hidden_states[:, :-1],
            num_iterations,
        )
    if fast_weight_gradients_only:
        torch.autograd.backward(
            hidden_states[:, :-1],
            hidden_gradient,
            inputs=(model.fast_weight_grad_buffer,),
        )
    else:
        hidden_states[:, :-1].backward(hidden_gradient)
    if update_state:
        with torch.no_grad(), autocast():
            embeddings = model.bidirectional_forward(
                hidden_states,
                attention_mask,
            )
        with autocast():
            model.update_state(embeddings, attention_mask)
    return loss


def second_pass(
    model,
    input_ids,
    assistant_mask,
    attention_mask,
    num_iterations,
    embedding_backpropagation="full",
):
    model.set_fast_weight_mode(FastWeightMLP.PLAIN)
    if embedding_backpropagation == "full":
        with autocast():
            propagated_embeddings = model.embedding_forward(
                input_ids,
                attention_mask,
            )
    elif embedding_backpropagation == "detached":
        with autocast():
            embedding_hidden_states = model.backbone_forward(
                input_ids=input_ids,
            )
            propagated_embeddings = model.bidirectional_forward(
                embedding_hidden_states.detach(),
                attention_mask,
            )
    elif embedding_backpropagation == "none":
        with torch.no_grad(), autocast():
            propagated_embeddings = model.embedding_forward(
                input_ids,
                attention_mask,
            )
    else:
        raise ValueError(
            "embedding_backpropagation must be full, detached, or none"
        )
    embeddings = propagated_embeddings.detach().requires_grad_(True)
    model.set_fast_weight_mode(FastWeightMLP.SECOND_PASS)
    double_input_ids = (
        input_ids[:, None].expand(-1, 2, -1).flatten(0, 1)
    )
    with autocast():
        loss_hidden_states = model.second_pass_forward(
            double_input_ids,
            embeddings,
            attention_mask,
            logits_to_keep=slice(0, -1),
        )
        _, hidden_gradient = loss_and_hidden_gradient(
            model,
            input_ids,
            assistant_mask,
            loss_hidden_states,
            num_iterations,
        )
    loss_hidden_states.backward(hidden_gradient)
    embedding_gradient = embeddings.grad.detach()
    model.set_fast_weight_mode(FastWeightMLP.PLAIN)
    if embedding_backpropagation != "none":
        (
            propagated_embeddings
            * embedding_gradient.to(propagated_embeddings.dtype)
        ).sum().backward()
    with autocast():
        model.update_state(
            embeddings,
            attention_mask,
            subtract_gradients=True,
        )


def classify_parameter(name: str) -> str:
    if "fast_log_lr" in name:
        return "fast_log_lr"
    if "fast_m" in name:
        return "fast_m"
    if "fast_p_" in name:
        return "fast_lr_projections"
    if any(key in name for key in ("up_fast", "gate_fast", "down_fast")):
        return "fast_mlp_projections"
    if "bidirectional_head" in name or "embedding_norm" in name:
        return "lr_embedding_network"
    if "self_attn" in name:
        return "slow_attention"
    if ".mlp." in name:
        return "slow_mlp"
    return "slow_norms"


@torch.no_grad()
def gradient_summary(model):
    families = {}
    layers = {}
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        square = grad.double().square().sum()
        total += square
        family = classify_parameter(name)
        entry = families.setdefault(
            family,
            {"square": torch.zeros_like(total), "count": 0, "max": 0.0},
        )
        entry["square"] += square
        entry["count"] += grad.numel()
        entry["max"] = max(entry["max"], grad.abs().max().item())
        if ".layers." in name:
            try:
                layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
            except ValueError:
                continue
            layers.setdefault(layer, torch.zeros_like(total)).add_(square)
    return {
        "total_norm": total.sqrt().item(),
        "families": {
            name: {
                "norm": value["square"].sqrt().item(),
                "rms": math.sqrt(value["square"].item() / value["count"]),
                "max": value["max"],
            }
            for name, value in families.items()
        },
        "backbone_layer_norms": {
            str(layer): value.sqrt().item()
            for layer, value in sorted(layers.items())
        },
    }


def materialize_records(records):
    return [
        {
            key: (
                value.float().item()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in record.items()
        }
        for record in records
    ]


def summarize_records(records):
    records = materialize_records(records)
    grouped = {}
    for record in records:
        key = (
            record["phase"],
            record["episode"],
            record["layer"],
        )
        grouped[key] = record
    return list(grouped.values())


def describe(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {}
    return {
        "min": values[0],
        "median": values[len(values) // 2],
        "max": values[-1],
    }


def compact_backward(records):
    records = summarize_records(records)
    first = [row for row in records if row["phase"] == "first"]
    second = [row for row in records if row["phase"] == "second"]
    first_by_layer = {}
    second_by_layer = {}
    second_by_episode = {}
    for layer in range(16):
        rows = [row for row in first if row["layer"] == layer]
        first_by_layer[str(layer)] = {
            "raw_rms": describe(row["raw_rms"] for row in rows),
            "output_grad_rms": describe(
                row["output_grad_rms"] for row in rows
            ),
        }
        rows = [row for row in second if row["layer"] == layer]
        second_by_layer[str(layer)] = {
            key: describe(row[key] for row in rows)
            for key in (
                "raw_rms",
                "future_rms",
                "future_to_raw",
                "raw_future_cosine",
                "inverse_rms",
                "lr_rms",
                "lr_max",
                "state_update_rms",
                "injected_activation_grad_norm",
                "injected_down_grad_norm",
                "lr_tensor_grad_norm",
                "example_raw_rms_min",
                "example_raw_rms_max",
            )
        }
    for episode in range(15):
        rows = [row for row in second if row["episode"] == episode]
        second_by_episode[str(episode)] = {
            "raw_rms": describe(row["raw_rms"] for row in rows),
            "future_rms": describe(row["future_rms"] for row in rows),
            "future_to_raw": describe(
                row["future_to_raw"] for row in rows
            ),
            "inverse_rms": describe(row["inverse_rms"] for row in rows),
            "example_raw_rms_min": describe(
                row["example_raw_rms_min"] for row in rows
            ),
            "example_raw_rms_max": describe(
                row["example_raw_rms_max"] for row in rows
            ),
            "injected_activation_grad_rss": math.sqrt(
                sum(
                    row["injected_activation_grad_norm"] ** 2
                    for row in rows
                )
            ),
            "injected_down_grad_rss": math.sqrt(
                sum(
                    row["injected_down_grad_norm"] ** 2
                    for row in rows
                )
            ),
            "lr_tensor_grad_rss": math.sqrt(
                sum(row["lr_tensor_grad_norm"] ** 2 for row in rows)
            ),
        }
    return {
        "first_by_layer": first_by_layer,
        "second_by_layer": second_by_layer,
        "second_by_episode": second_by_episode,
    }


def compact_updates(records):
    records = summarize_records(records)
    first = [row for row in records if row["phase"] == "first"]
    by_episode = {}
    by_layer = {}
    for episode in range(15):
        rows = [row for row in first if row["episode"] == episode]
        by_episode[str(episode)] = {
            key: describe(row[key] for row in rows)
            for key in (
                "raw_rms",
                "lr_rms",
                "lr_max",
                "update_rms",
                "state_rms_before",
            )
        }
    for layer in range(16):
        rows = [row for row in first if row["layer"] == layer]
        by_layer[str(layer)] = {
            key: describe(row[key] for row in rows)
            for key in (
                "raw_rms",
                "lr_rms",
                "lr_max",
                "update_rms",
                "state_rms_before",
            )
        }
    return {"by_episode": by_episode, "by_layer": by_layer}


def countdown_audit(records):
    by_key = {
        (row["phase"], row["episode"], row["layer"]): row
        for row in records
    }
    results = {}
    for layer in PROBE.trace_layers:
        first_raw = [
            by_key[("first", episode, layer)]["raw"]
            for episode in range(16)
        ]
        rows = []
        for episode in range(15):
            replay = by_key[("second", episode, layer)]
            expected = torch.stack(
                first_raw[episode + 1 :]
            ).sum(dim=0)
            observed = replay["future"]
            error = observed - expected
            expected_norm = expected.norm()
            observed_norm = observed.norm()
            first_current = first_raw[episode]
            replay_current = replay["raw"]
            rows.append(
                {
                    "episode": episode,
                    "future_relative_error": (
                        error.norm()
                        / expected_norm.clamp_min(1.0e-30)
                    ).item(),
                    "future_cosine": (
                        (expected * observed).sum()
                        / (
                            expected_norm
                            * observed_norm
                        ).clamp_min(1.0e-30)
                    ).item(),
                    "observed_over_expected_norm": (
                        observed_norm
                        / expected_norm.clamp_min(1.0e-30)
                    ).item(),
                    "current_replay_relative_error": (
                        (replay_current - first_current).norm()
                        / first_current.norm().clamp_min(1.0e-30)
                    ).item(),
                }
            )
        results[str(layer)] = rows
    return results


def trajectory_audit(records):
    by_key = {
        (row["phase"], row["episode"], row["layer"]): row
        for row in records
    }
    results = {}
    for layer in PROBE.trace_layers:
        rows = []
        for episode in range(15):
            first = by_key[("first", episode, layer)]
            replay = by_key[("second", episode, layer)]
            row = {"episode": episode}
            for key in (
                "state_before",
                "raw",
                "learning_rate",
                "update",
            ):
                expected = first[key]
                observed = replay[key]
                expected_norm = expected.norm()
                row[f"{key}_relative_error"] = (
                    (observed - expected).norm()
                    / expected_norm.clamp_min(1.0e-30)
                ).item()
            rows.append(row)
        results[str(layer)] = rows
    return results


def load_state(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key.replace("_orig_mod.", ""): value
        for key, value in state.items()
    }


def load_model(
    config_path: Path,
    state_path: Path,
    base_state_path: Path | None,
    runtime_batch_size: int = 1,
    attention_kernel: str | None = "gpu_flash_attention",
):
    config = OmegaConf.load(config_path)
    config.attention_kernel = attention_kernel
    torch.manual_seed(42)
    model = FoItttModel(config)
    state = load_state(base_state_path or state_path)
    model.load_state_dict(
        state,
        strict=base_state_path is None,
    )
    del state
    gc.collect()
    model.to("cuda")
    model.lm_head.to(torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "embed_tokens" not in name and "lm_head" not in name:
            parameter.requires_grad_(True)
    model.init_state(runtime_batch_size, torch.device("cuda"))
    model.zero_grad(set_to_none=False)
    return model, config


def snapshot_gradients(model):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def run_probe(
    label,
    model,
    config,
    batch,
    retain_gradients=False,
    inject_activation=True,
    embedding_backpropagation="full",
    trace_layers=(),
    use_reference_replay=False,
    use_reference_local_jacobian=False,
    meta_grad_eps=None,
    detach_meta_rms=False,
    future_gradient_mode="full",
):
    PROBE.backward = []
    PROBE.updates = []
    PROBE.forwards = []
    PROBE.raw_tensors = []
    PROBE.update_tensors = []
    PROBE.trace_layers = tuple(trace_layers)
    PROBE.use_reference_replay = use_reference_replay
    PROBE.reference_raw = {}
    PROBE.use_reference_local_jacobian = (
        use_reference_local_jacobian
    )
    PROBE.reference_local = {}
    PROBE.meta_grad_eps = meta_grad_eps
    PROBE.detach_meta_rms = detach_meta_rms
    PROBE.future_gradient_mode = future_gradient_mode
    PROBE.inject_activation = inject_activation
    handles = install_probes(model)
    episodes = tuple(
        zip(
            batch["input_ids"].unbind(1),
            batch["assistant_mask"].unbind(1),
            batch["attention_mask"].unbind(1),
        )
    )
    losses = []
    model.empty_state()
    for index, episode in enumerate(episodes):
        PROBE.phase = "first"
        PROBE.episode = index
        loss = first_pass(
            model,
            *episode,
            config.trainer_num_logit_iterations,
            update_state=index != len(episodes) - 1,
        )
        losses.append(loss.item())

    model.accumulate_gradients()
    model.finalize_gradients()
    first_final_norms = (
        model.fast_weight_final_grad_norm.detach().cpu().tolist()
    )
    model.zero_grad(set_to_none=False)

    for index, episode in enumerate(episodes[:-1]):
        PROBE.phase = "second"
        PROBE.episode = index
        second_pass(
            model,
            *episode,
            config.trainer_num_logit_iterations,
            embedding_backpropagation=embedding_backpropagation,
        )

    PROBE.phase = "terminal"
    PROBE.episode = len(episodes) - 1
    first_pass(
        model,
        *episodes[-1],
        config.trainer_num_logit_iterations,
        update_state=False,
        fast_weight_gradients_only=False,
    )
    model.accumulate_gradients(subtract=True)
    replay_error_by_layer = [
        module.relative_grad_error(buffer, norm).item()
        for module, _, buffer, _, norm in model._iter_fast_weight_runtime()
    ]
    grads = gradient_summary(model)

    for handle in handles:
        handle.remove()
    result = {
        "label": label,
        "losses": losses,
        "mean_loss": sum(losses) / len(losses),
        "first_final_fast_gradient_norms": first_final_norms,
        "replay_error_by_layer": replay_error_by_layer,
        "gradient_summary": grads,
        "backward": compact_backward(PROBE.backward),
        "updates": compact_updates(PROBE.updates),
        "forwards": summarize_records(PROBE.forwards),
        "countdown_audit": countdown_audit(PROBE.raw_tensors),
        "trajectory_audit": trajectory_audit(PROBE.update_tensors),
    }
    gradient_snapshot = (
        snapshot_gradients(model) if retain_gradients else None
    )
    PROBE.reference_raw = {}
    PROBE.reference_local = {}
    model.empty_state()
    model.zero_grad(set_to_none=True)
    return result, gradient_snapshot


def direct_gradient(model, config, batch):
    """Differentiate every horizon loss while detaching fast-state updates."""
    episodes = tuple(
        zip(
            batch["input_ids"].unbind(1),
            batch["assistant_mask"].unbind(1),
            batch["attention_mask"].unbind(1),
        )
    )
    model.empty_state()
    model.zero_grad(set_to_none=True)
    losses = []
    for index, episode in enumerate(episodes):
        PROBE.phase = "direct"
        PROBE.episode = index
        loss = first_pass(
            model,
            *episode,
            config.trainer_num_logit_iterations,
            update_state=index != len(episodes) - 1,
            fast_weight_gradients_only=False,
        )
        losses.append(loss.item())
    return losses, gradient_summary(model)


def fast_state_objective(model, config, batch):
    """Evaluate the actual horizon objective while constructing fast states."""
    episodes = tuple(
        zip(
            batch["input_ids"].unbind(1),
            batch["assistant_mask"].unbind(1),
            batch["attention_mask"].unbind(1),
        )
    )
    model.empty_state()
    model.zero_grad(set_to_none=True)
    losses = []
    for index, episode in enumerate(episodes):
        PROBE.phase = "finite_difference"
        PROBE.episode = index
        loss = first_pass(
            model,
            *episode,
            config.trainer_num_logit_iterations,
            update_state=index != len(episodes) - 1,
            fast_weight_gradients_only=True,
        )
        losses.append(loss.item())
    model.empty_state()
    model.zero_grad(set_to_none=True)
    return sum(losses)


@torch.no_grad()
def perturb_parameters(model, direction, scale):
    for name, parameter in model.named_parameters():
        if name in direction:
            parameter.add_(
                direction[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                alpha=scale,
            )


def normalized_slow_meta_direction(
    model,
    full_gradients,
    direct_gradients,
):
    direction = {}
    square = 0.0
    for name, parameter in model.named_parameters():
        if classify_parameter(name) not in (
            "slow_attention",
            "slow_mlp",
            "slow_norms",
        ):
            continue
        reference = full_gradients.get(
            name,
            direct_gradients.get(name),
        )
        if reference is None:
            continue
        meta = (
            full_gradients.get(name, torch.zeros_like(reference))
            - direct_gradients.get(name, torch.zeros_like(reference))
        ).float()
        direction[name] = meta
        square += meta.double().square().sum().item()
    norm = math.sqrt(square)
    for name in direction:
        direction[name].div_(max(norm, 1.0e-30))
    predicted_full = predicted_direct = 0.0
    for name, unit in direction.items():
        predicted_full += (
            full_gradients[name].float() * unit
        ).sum().item()
        predicted_direct += (
            direct_gradients.get(
                name,
                torch.zeros_like(unit),
            ).float()
            * unit
        ).sum().item()
    return direction, {
        "meta_norm": norm,
        "fo_full_directional_derivative": predicted_full,
        "detached_state_directional_derivative": predicted_direct,
        "fo_meta_directional_derivative": (
            predicted_full - predicted_direct
        ),
    }


def one_episode_gradient(model, config, batch, target_episode):
    """Differentiate one loss after constructing its preceding fast state."""
    episodes = tuple(
        zip(
            batch["input_ids"].unbind(1),
            batch["assistant_mask"].unbind(1),
            batch["attention_mask"].unbind(1),
        )
    )
    model.empty_state()
    model.zero_grad(set_to_none=True)
    losses = []
    for index, episode in enumerate(episodes[: target_episode + 1]):
        PROBE.phase = "episode_gradient"
        PROBE.episode = index
        loss = first_pass(
            model,
            *episode,
            config.trainer_num_logit_iterations,
            update_state=index != target_episode,
            fast_weight_gradients_only=index != target_episode,
        )
        losses.append(loss.item())
    return losses[-1], gradient_summary(model)


def add_comparison(groups, key, full, direct):
    full = full.detach().float().cpu()
    direct = direct.detach().float().cpu()
    entry = groups.setdefault(
        key,
        {
            "full2": 0.0,
            "direct2": 0.0,
            "dot": 0.0,
            "difference2": 0.0,
        },
    )
    entry["full2"] += full.square().sum().item()
    entry["direct2"] += direct.square().sum().item()
    entry["dot"] += (full * direct).sum().item()
    entry["difference2"] += (full - direct).square().sum().item()


def finish_comparison(groups):
    result = {}
    for key, value in groups.items():
        full_norm = math.sqrt(value["full2"])
        direct_norm = math.sqrt(value["direct2"])
        difference_norm = math.sqrt(value["difference2"])
        denominator = max(full_norm * direct_norm, 1e-30)
        result[key] = {
            "full_norm": full_norm,
            "direct_norm": direct_norm,
            "fo_correction_norm": difference_norm,
            "full_direct_cosine": value["dot"] / denominator,
            "full_over_direct_norm": full_norm / max(direct_norm, 1e-30),
            "fo_over_direct_norm": (
                difference_norm / max(direct_norm, 1e-30)
            ),
            "full_projection_on_direct": (
                value["dot"] / max(value["direct2"], 1e-30)
            ),
            "fo_direct_cosine": (
                (value["dot"] - value["direct2"])
                / max(difference_norm * direct_norm, 1e-30)
            ),
        }
    return result


def compare_full_and_direct(model, full_gradients):
    families = {}
    layers = {}
    parameters = []
    for name, parameter in model.named_parameters():
        if name not in full_gradients:
            continue
        direct = (
            parameter.grad.detach()
            if parameter.grad is not None
            else torch.zeros_like(parameter)
        )
        full = full_gradients[name]
        add_comparison(
            families,
            classify_parameter(name),
            full,
            direct,
        )
        if ".layers." in name:
            layer = name.split(".layers.", 1)[1].split(".", 1)[0]
            add_comparison(layers, layer, full, direct)

        full_float = full.float()
        direct_float = direct.detach().float().cpu()
        full_norm = full_float.norm().item()
        direct_norm = direct_float.norm().item()
        difference_norm = (full_float - direct_float).norm().item()
        parameters.append(
            {
                "name": name,
                "full_norm": full_norm,
                "direct_norm": direct_norm,
                "fo_correction_norm": difference_norm,
                "full_direct_cosine": (
                    (full_float * direct_float).sum().item()
                    / max(full_norm * direct_norm, 1e-30)
                ),
                "fo_over_direct_norm": (
                    difference_norm / max(direct_norm, 1e-30)
                ),
            }
        )
    return {
        "families": finish_comparison(families),
        "backbone_layers": finish_comparison(layers),
        "largest_fo_corrections": sorted(
            parameters,
            key=lambda row: row["fo_correction_norm"],
            reverse=True,
        )[:20],
    }


def compare_gradient_snapshots(full_gradients, direct_gradients):
    families = {}
    layers = {}
    for name in full_gradients.keys() | direct_gradients.keys():
        reference = full_gradients.get(name, direct_gradients.get(name))
        full = full_gradients.get(name, torch.zeros_like(reference))
        direct = direct_gradients.get(name, torch.zeros_like(reference))
        add_comparison(
            families,
            classify_parameter(name),
            full,
            direct,
        )
        if ".layers." in name:
            layer = name.split(".layers.", 1)[1].split(".", 1)[0]
            add_comparison(layers, layer, full, direct)
    return {
        "families": finish_comparison(families),
        "backbone_layers": finish_comparison(layers),
    }


def slow_gradient_intervention_comparison(
    full_gradients,
    intervention_gradients,
):
    family_groups = {}
    layer_groups = {}
    parameters = []
    selected_layers = {0, 7, 15}
    selected_suffixes = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "post_attention_layernorm.weight",
    )
    for name in full_gradients.keys() | intervention_gradients.keys():
        family = classify_parameter(name)
        if family not in (
            "slow_attention",
            "slow_mlp",
            "slow_norms",
        ):
            continue
        reference = full_gradients.get(
            name,
            intervention_gradients.get(name),
        )
        full = full_gradients.get(
            name,
            torch.zeros_like(reference),
        ).float()
        intervention = intervention_gradients.get(
            name,
            torch.zeros_like(reference),
        ).float()
        add_comparison(
            family_groups,
            family,
            full,
            intervention,
        )
        layer = None
        if ".layers." in name:
            layer = int(
                name.split(".layers.", 1)[1].split(".", 1)[0]
            )
            add_comparison(
                layer_groups,
                str(layer),
                full,
                intervention,
            )
        full_norm = full.norm().item()
        intervention_norm = intervention.norm().item()
        difference_norm = (full - intervention).norm().item()
        parameters.append(
            {
                "name": name,
                "full_norm": full_norm,
                "intervention_norm": intervention_norm,
                "removed_norm": difference_norm,
                "removed_over_full": (
                    difference_norm / max(full_norm, 1.0e-30)
                ),
                "intervention_over_full": (
                    intervention_norm / max(full_norm, 1.0e-30)
                ),
                "cosine": (
                    (full * intervention).sum().item()
                    / max(
                        full_norm * intervention_norm,
                        1.0e-30,
                    )
                ),
                "selected": (
                    layer in selected_layers
                    and name.endswith(selected_suffixes)
                ),
            }
        )
    return {
        "families": finish_comparison(family_groups),
        "backbone_layers": finish_comparison(layer_groups),
        "selected_parameters": [
            row for row in parameters if row["selected"]
        ],
        "largest_changes": sorted(
            parameters,
            key=lambda row: row["removed_norm"],
            reverse=True,
        )[:20],
        "largest_relative_changes": sorted(
            parameters,
            key=lambda row: row["removed_over_full"],
            reverse=True,
        )[:20],
    }


def slow_route_reconstruction(
    full_gradients,
    no_future_gradients,
    no_lr_gradients,
    no_fast_gradients,
):
    groups = {}
    for name in (
        full_gradients.keys()
        | no_future_gradients.keys()
        | no_lr_gradients.keys()
        | no_fast_gradients.keys()
    ):
        family = classify_parameter(name)
        if family not in (
            "slow_attention",
            "slow_mlp",
            "slow_norms",
        ):
            continue
        reference = full_gradients.get(
            name,
            no_future_gradients.get(
                name,
                no_lr_gradients.get(name, no_fast_gradients.get(name)),
            ),
        )
        full = full_gradients.get(
            name, torch.zeros_like(reference)
        ).float()
        direct = no_future_gradients.get(
            name, torch.zeros_like(reference)
        ).float()
        no_lr = no_lr_gradients.get(
            name, torch.zeros_like(reference)
        ).float()
        no_fast = no_fast_gradients.get(
            name, torch.zeros_like(reference)
        ).float()
        lr_route = full - no_lr
        fast_route = full - no_fast
        total_extra = full - direct
        residual = total_extra - lr_route - fast_route
        entry = groups.setdefault(
            family,
            {
                "full2": 0.0,
                "direct2": 0.0,
                "total_extra2": 0.0,
                "lr_route2": 0.0,
                "fast_route2": 0.0,
                "residual2": 0.0,
                "lr_fast_dot": 0.0,
            },
        )
        entry["full2"] += full.square().sum().item()
        entry["direct2"] += direct.square().sum().item()
        entry["total_extra2"] += total_extra.square().sum().item()
        entry["lr_route2"] += lr_route.square().sum().item()
        entry["fast_route2"] += fast_route.square().sum().item()
        entry["residual2"] += residual.square().sum().item()
        entry["lr_fast_dot"] += (lr_route * fast_route).sum().item()
    result = {}
    for family, values in groups.items():
        lr_norm = math.sqrt(values["lr_route2"])
        fast_norm = math.sqrt(values["fast_route2"])
        total_extra_norm = math.sqrt(values["total_extra2"])
        result[family] = {
            "full_norm": math.sqrt(values["full2"]),
            "no_future_norm": math.sqrt(values["direct2"]),
            "total_future_extra_norm": total_extra_norm,
            "lr_route_norm": lr_norm,
            "fast_projection_route_norm": fast_norm,
            "route_reconstruction_residual_norm": math.sqrt(
                values["residual2"]
            ),
            "extra_over_no_future": (
                total_extra_norm
                / max(math.sqrt(values["direct2"]), 1.0e-30)
            ),
            "lr_fast_route_cosine": (
                values["lr_fast_dot"]
                / max(lr_norm * fast_norm, 1.0e-30)
            ),
        }
    return result


@torch.no_grad()
def compare_muon_directions(model, full_gradients):
    """Compare current-gradient Muon directions on representative slow matrices."""
    groups = {}
    parameters = []
    for name, parameter in model.named_parameters():
        if (
            name not in full_gradients
            or parameter.grad is None
            or parameter.ndim != 2
            or getattr(parameter, "no_muon", False)
            or classify_parameter(name)
            not in ("slow_attention", "slow_mlp")
        ):
            continue
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        if (
            layer not in (0, 1)
            and not name.endswith(".mlp.down_proj.weight")
        ):
            continue
        full = full_gradients[name].to(
            device=parameter.device,
            dtype=torch.bfloat16,
        )
        direct = parameter.grad.detach().to(torch.bfloat16)
        full_update = newton_schulz(full, steps=5, eps=1e-6).float()
        direct_update = newton_schulz(
            direct,
            steps=5,
            eps=1e-6,
        ).float()
        add_comparison(
            groups,
            classify_parameter(name),
            full_update,
            direct_update,
        )
        full_norm = full_update.norm().item()
        direct_norm = direct_update.norm().item()
        parameters.append(
            {
                "name": name,
                "cosine": (
                    (full_update * direct_update).sum().item()
                    / max(full_norm * direct_norm, 1e-30)
                ),
                "relative_difference": (
                    (full_update - direct_update).norm().item()
                    / max(direct_norm, 1e-30)
                ),
            }
        )
        del full, direct, full_update, direct_update
    return {
        "groups": finish_comparison(groups),
        "parameters": parameters,
    }


def load_batch(batch_size=1, skip_rows=0):
    dataset = datasets.load_dataset(
        "aklein4/horizons-10B",
        split="train",
        streaming=True,
    )
    iterator = iter(dataset)
    for _ in range(skip_rows):
        next(iterator)
    rows = [next(iterator) for _ in range(batch_size)]
    collator = HorizonCollator(
        "meta-llama/Llama-3.2-1B-Instruct",
        max_length=1024,
        cluster_length=16,
    )
    batch = collator(rows)
    return {
        key: value.to("cuda")
        for key, value in batch.items()
    }


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
        default=["fresh", "step100", "step200"],
    )
    parser.add_argument(
        "--compare-direct",
        action="store_true",
        help="Compare the full FO gradient with detached-state loss gradients.",
    )
    parser.add_argument(
        "--compare-muon",
        action="store_true",
        help="Also compare Newton-Schulz directions for representative matrices.",
    )
    parser.add_argument(
        "--episode-conflict",
        action="store_true",
        help="Compare the summed detached-state gradient with episodes 0 and 15.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument(
        "--path-ablation",
        action="store_true",
        help="Separate causal-backbone activation and LR-embedding meta paths.",
    )
    parser.add_argument(
        "--countdown-audit",
        action="store_true",
        help="Store selected raw gradients and compare countdown suffixes.",
    )
    parser.add_argument(
        "--reference-replay-ablation",
        action="store_true",
        help="Compare current replay with first-pass-value replay.",
    )
    parser.add_argument(
        "--reference-local-jacobian-ablation",
        action="store_true",
        help=(
            "Compare replay Jacobians with local activations and output "
            "gradients captured during the first pass."
        ),
    )
    parser.add_argument(
        "--normalization-ablation",
        action="store_true",
        help=(
            "Compare the current differentiated RMS normalizer with a "
            "detached denominator and larger meta-gradient epsilons."
        ),
    )
    parser.add_argument(
        "--finite-difference-meta",
        action="store_true",
        help=(
            "Check the slow FO correction against central differences of "
            "the actual fast-state horizon objective."
        ),
    )
    parser.add_argument(
        "--future-gradient-interventions",
        action="store_true",
        help=(
            "Compare full FO gradients with zero future gradient, detached "
            "future LR, and detached future fast-projection routes."
        ),
    )
    parser.add_argument(
        "--deterministic-attention",
        action="store_true",
        help=(
            "Use the ordinary attention implementation so repeated backward "
            "passes can be compared without flash-attention nondeterminism."
        ),
    )
    args = parser.parse_args()
    batch = load_batch(args.batch_size, args.skip_rows)
    print(
        json.dumps(
            {
                "attention_tokens": batch["attention_mask"].sum(-1).tolist(),
                "assistant_tokens": batch["assistant_mask"].sum(-1).tolist(),
            }
        ),
        flush=True,
    )

    checkpoint_specs = {
        "fresh": (
            args.checkpoint_root / "fo-100/000000000100/config.json",
            args.checkpoint_root / "fo-100/000000000100/model.pt",
            args.checkpoint_root
            / "llama-base/000000000000/model.pt",
        ),
        "step100": (
            args.checkpoint_root / "fo-100/000000000100/config.json",
            args.checkpoint_root / "fo-100/000000000100/model.pt",
            None,
        ),
        "step200": (
            args.checkpoint_root / "fo-200/000000000200/config.json",
            args.checkpoint_root / "fo-200/000000000200/model.pt",
            None,
        ),
    }
    for label in args.labels:
        config_path, state_path, base_path = checkpoint_specs[label]
        model, model_config = load_model(
            config_path,
            state_path,
            base_path,
            runtime_batch_size=args.batch_size,
            attention_kernel=(
                None
                if args.deterministic_attention
                else "gpu_flash_attention"
            ),
        )
        # Chunking the vocabulary projection is mathematically neutral.  The
        # TPU run used four chunks because its global batch made 1023 loss
        # tokens divisible by four; this single-example GPU probe uses one.
        model_config.trainer_num_logit_iterations = 1
        if args.future_gradient_interventions:
            snapshots = {}
            summaries = {}
            for variant, future_gradient_mode in (
                ("full", "full"),
                ("no_future", "zero"),
                ("detach_lr", "detach_lr"),
                ("detach_fast_projections", "detach_fast"),
            ):
                variant_result, variant_gradients = run_probe(
                    label,
                    model,
                    model_config,
                    batch,
                    retain_gradients=True,
                    future_gradient_mode=future_gradient_mode,
                )
                snapshots[variant] = variant_gradients
                summaries[variant] = {
                    "gradient_summary": variant_result[
                        "gradient_summary"
                    ],
                    "replay_error_by_layer": variant_result[
                        "replay_error_by_layer"
                    ],
                }
            full = snapshots["full"]
            comparisons = {
                variant: slow_gradient_intervention_comparison(
                    full,
                    snapshots[variant],
                )
                for variant in (
                    "no_future",
                    "detach_lr",
                    "detach_fast_projections",
                )
            }
            print(
                json.dumps(
                    {
                        "label": label,
                        "summaries": summaries,
                        "full_vs_intervention": comparisons,
                        "route_reconstruction": (
                            slow_route_reconstruction(
                                full,
                                snapshots["no_future"],
                                snapshots["detach_lr"],
                                snapshots[
                                    "detach_fast_projections"
                                ],
                            )
                        ),
                    }
                ),
                flush=True,
            )
            del snapshots
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if args.finite_difference_meta:
            full_result, full_gradients = run_probe(
                label,
                model,
                model_config,
                batch,
                retain_gradients=True,
            )
            direct_losses, _ = direct_gradient(
                model,
                model_config,
                batch,
            )
            direct_gradients = snapshot_gradients(model)
            direction, predictions = (
                normalized_slow_meta_direction(
                    model,
                    full_gradients,
                    direct_gradients,
                )
            )
            epsilon = 0.05
            perturb_parameters(model, direction, epsilon)
            plus = fast_state_objective(
                model,
                model_config,
                batch,
            )
            perturb_parameters(model, direction, -2.0 * epsilon)
            minus = fast_state_objective(
                model,
                model_config,
                batch,
            )
            perturb_parameters(model, direction, epsilon)
            center = sum(direct_losses)
            print(
                json.dumps(
                    {
                        "label": label,
                        "epsilon": epsilon,
                        "objective": {
                            "minus": minus,
                            "center": center,
                            "plus": plus,
                            "central_directional_derivative": (
                                (plus - minus) / (2.0 * epsilon)
                            ),
                            "central_second_difference": (
                                (plus - 2.0 * center + minus)
                                / (epsilon * epsilon)
                            ),
                        },
                        "predictions": predictions,
                        "full_gradient_norm": full_result[
                            "gradient_summary"
                        ]["total_norm"],
                    }
                ),
                flush=True,
            )
            del direction, full_gradients, direct_gradients
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if args.reference_local_jacobian_ablation:
            current_result, current_gradients = run_probe(
                label,
                model,
                model_config,
                batch,
                retain_gradients=True,
            )
            reference_result, reference_gradients = run_probe(
                label,
                model,
                model_config,
                batch,
                retain_gradients=True,
                use_reference_replay=True,
                use_reference_local_jacobian=True,
            )
            print(
                json.dumps(
                    {
                        "label": label,
                        "current_gradient_summary": current_result[
                            "gradient_summary"
                        ],
                        "reference_gradient_summary": reference_result[
                            "gradient_summary"
                        ],
                        "current_replay_error": current_result[
                            "replay_error_by_layer"
                        ],
                        "reference_replay_error": reference_result[
                            "replay_error_by_layer"
                        ],
                        "current_vs_reference": (
                            compare_gradient_snapshots(
                                current_gradients,
                                reference_gradients,
                            )
                        ),
                    }
                ),
                flush=True,
            )
            del current_gradients, reference_gradients
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if args.normalization_ablation:
            variants = {}
            snapshots = {}
            for (
                variant,
                meta_grad_eps,
                detach_meta_rms,
            ) in (
                ("current", None, False),
                ("eps_1e-8", 1.0e-8, False),
                ("eps_1e-6", 1.0e-6, False),
                ("detached_rms", None, True),
            ):
                variant_result, variant_gradients = run_probe(
                    label,
                    model,
                    model_config,
                    batch,
                    retain_gradients=True,
                    meta_grad_eps=meta_grad_eps,
                    detach_meta_rms=detach_meta_rms,
                )
                variants[variant] = {
                    "gradient_summary": variant_result[
                        "gradient_summary"
                    ],
                    "replay_error_by_layer": variant_result[
                        "replay_error_by_layer"
                    ],
                    "backward": variant_result["backward"],
                }
                snapshots[variant] = variant_gradients
            current = snapshots["current"]
            comparisons = {
                variant: compare_gradient_snapshots(
                    current,
                    snapshot,
                )
                for variant, snapshot in snapshots.items()
                if variant != "current"
            }
            print(
                json.dumps(
                    {
                        "label": label,
                        "variants": variants,
                        "current_vs_variant": comparisons,
                    }
                ),
                flush=True,
            )
            del snapshots
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if args.reference_replay_ablation:
            current_result, current_gradients = run_probe(
                label,
                model,
                model_config,
                batch,
                retain_gradients=True,
            )
            reference_result, reference_gradients = run_probe(
                label,
                model,
                model_config,
                batch,
                retain_gradients=True,
                use_reference_replay=True,
            )
            print(
                json.dumps(
                    {
                        "label": label,
                        "current_gradient_summary": current_result[
                            "gradient_summary"
                        ],
                        "reference_gradient_summary": reference_result[
                            "gradient_summary"
                        ],
                        "current_replay_error": current_result[
                            "replay_error_by_layer"
                        ],
                        "reference_replay_error": reference_result[
                            "replay_error_by_layer"
                        ],
                        "current_vs_reference": (
                            compare_gradient_snapshots(
                                current_gradients,
                                reference_gradients,
                            )
                        ),
                    }
                ),
                flush=True,
            )
            del current_gradients, reference_gradients
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if args.path_ablation:
            direct_losses, direct_summary = direct_gradient(
                model,
                model_config,
                batch,
            )
            direct_gradients = snapshot_gradients(model)
            model.empty_state()
            model.zero_grad(set_to_none=True)
            variants = {}
            for variant, inject_activation, embedding_mode in (
                ("current", True, "full"),
                ("activation_only", True, "none"),
                ("embedding_only", False, "full"),
                ("reference_detached", False, "detached"),
            ):
                variant_result, variant_gradients = run_probe(
                    label,
                    model,
                    model_config,
                    batch,
                    retain_gradients=True,
                    inject_activation=inject_activation,
                    embedding_backpropagation=embedding_mode,
                )
                variants[variant] = {
                    "gradient_summary": variant_result[
                        "gradient_summary"
                    ],
                    "versus_direct": compare_gradient_snapshots(
                        variant_gradients,
                        direct_gradients,
                    ),
                }
                del variant_gradients
            print(
                json.dumps(
                    {
                        "label": label,
                        "direct_losses": direct_losses,
                        "direct_gradient_summary": direct_summary,
                        "variants": variants,
                    }
                ),
                flush=True,
            )
            del direct_gradients
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if args.episode_conflict:
            direct_losses, direct_summary = direct_gradient(
                model,
                model_config,
                batch,
            )
            summed_gradients = snapshot_gradients(model)
            episode_comparisons = {}
            for episode_index in (0, 15):
                episode_loss, episode_summary = one_episode_gradient(
                    model,
                    model_config,
                    batch,
                    episode_index,
                )
                episode_comparisons[str(episode_index)] = {
                    "loss": episode_loss,
                    "gradient_summary": episode_summary,
                    "summed_vs_episode": compare_full_and_direct(
                        model,
                        summed_gradients,
                    ),
                }
            print(
                json.dumps(
                    {
                        "label": label,
                        "direct_losses": direct_losses,
                        "summed_gradient_summary": direct_summary,
                        "episode_comparisons": episode_comparisons,
                    }
                ),
                flush=True,
            )
            del summed_gradients
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue
        result, full_gradients = run_probe(
            label,
            model,
            model_config,
            batch,
            retain_gradients=args.compare_direct,
            trace_layers=(0, 1, 15) if args.countdown_audit else (),
        )
        if args.compare_direct:
            direct_losses, direct_summary = direct_gradient(
                model,
                model_config,
                batch,
            )
            comparison = compare_full_and_direct(
                model,
                full_gradients,
            )
            muon_comparison = (
                compare_muon_directions(model, full_gradients)
                if args.compare_muon
                else None
            )
            compact_result = {
                "label": label,
                "full_losses": result["losses"],
                "direct_losses": direct_losses,
                "full_gradient_summary": result["gradient_summary"],
                "direct_gradient_summary": direct_summary,
                "comparison": comparison,
                "muon_comparison": muon_comparison,
            }
            print(json.dumps(compact_result), flush=True)
            del full_gradients
        else:
            print(json.dumps(result), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
