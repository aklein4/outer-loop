"""Create self-contained OLoop-LoRA fast-weight prediction visualizations.

The model is always constructed from the Hydra-composed fresh config. Its
learning rate and checkpoint can be overridden independently::

    python scripts/visualize_oloop_lora.py \
      base_lr=1e-4 \
      checkpoint_url=aklein4/my-oloop-lora \
      checkpoint_step=200 \
      sampling.n=3 name=oloop_lora_step200

Sampling options match ``visualize_piano_token_gates.py``. This architecture
has no token gates, so the table contains prediction diagnostics only.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator
from models import load_checkpoint_state
from scripts.visualize_piano_token_gates import (
    Trajectory,
    autocast_context,
    empty_fast_weights,
    filename_stem,
    loss_and_hidden_gradient,
    model_url_folder,
    prediction_statistics,
    sample_trajectories,
    write_episode_data,
    write_outputs,
)
from utils import constants
from utils.import_utils import import_model
from utils.torch_modules import enable_gradient_checkpointing


def resolved_model_config(config: DictConfig) -> DictConfig:
    model_config = OmegaConf.create(
        OmegaConf.to_container(config.model, resolve=True)
    )
    if config.base_lr is not None:
        model_config.base_lr = float(config.base_lr)
    if config.checkpoint_url is not None:
        model_config.pretrained_url = str(config.checkpoint_url)
    if config.checkpoint_step is not None:
        model_config.pretrained_step = int(config.checkpoint_step)
    return model_config


def load_model(config: DictConfig, device: torch.device):
    model_config = resolved_model_config(config)
    if not str(model_config.type).endswith("oloop_lora.OLoopLoRAModel"):
        raise ValueError(
            "This visualizer requires an oloop_lora.OLoopLoRAModel config; "
            f"got {model_config.type!r}"
        )

    model = import_model(model_config.type)(model_config)
    # Match evaluate_icl's fresh-config path: move to CUDA before checkpoint
    # loading so loading a base Llama checkpoint performs LoRA SVD on the GPU.
    model = model.float().to(device)
    if model_config.pretrained_url is not None:
        if model_config.pretrained_step is None:
            raise ValueError(
                "checkpoint_step (or model.pretrained_step) is required when "
                "a checkpoint URL is configured"
            )
        print(
            f"Loading {model_config.pretrained_url} at step "
            f"{model_config.pretrained_step} with "
            f"strict={bool(model_config.pretrained_strict)}",
            flush=True,
        )
        model = load_checkpoint_state(
            model,
            str(model_config.pretrained_url),
            int(model_config.pretrained_step),
            strict=bool(model_config.pretrained_strict),
        )

    model.train()
    if config.runtime.compute_dtype != "float32":
        model.lm_head.to(getattr(torch, str(config.runtime.compute_dtype)))
    enable_gradient_checkpointing(
        model, bool(config.runtime.gradient_checkpointing)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_config


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
    entries: list[dict[str, Any]] = []
    batch_size = int(config.runtime.batch_size)

    for batch_start in range(0, len(trajectories), batch_size):
        selected = trajectories[batch_start : batch_start + batch_size]
        batch = collator([trajectory.row for trajectory in selected])
        input_ids, assistant_mask, valid_mask = (
            batch[key].to(device)
            for key in ("input_ids", "assistant_mask", "attention_mask")
        )
        model.init_state(len(selected), device)
        if config.sampling.mode == "episodes":
            record_positions = [
                {int(trajectory.target_episode)} for trajectory in selected
            ]
        else:
            record_positions = [
                set(range(int(config.data.episode_count))) for _ in selected
            ]
        maximum_position = max(max(positions) for positions in record_positions)

        for position in range(maximum_position + 1):
            ids = input_ids[:, position]
            assistant = assistant_mask[:, position]
            valid = valid_mask[:, position]
            should_record = any(
                position in positions for positions in record_positions
            )

            empty_states = None
            if should_record:
                with empty_fast_weights(model), torch.no_grad(), autocast_context(
                    device, str(config.runtime.compute_dtype)
                ):
                    empty_states, _ = model(
                        input_ids=ids,
                        shift_states=True,
                        compute_logits=False,
                    )

            with autocast_context(device, str(config.runtime.compute_dtype)):
                updated_states, _ = model(
                    input_ids=ids,
                    shift_states=True,
                    compute_logits=False,
                )

            if should_record:
                with autocast_context(device, str(config.runtime.compute_dtype)):
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
                    prediction_count = max(
                        0, int(valid[batch_index].sum()) - 1
                    )
                    entry = write_episode_data(
                        output_root,
                        trajectory,
                        position,
                        collator.tokenizer,
                        ids,
                        assistant,
                        valid,
                        np.empty((0, prediction_count), dtype=np.float32),
                        statistics,
                        batch_index,
                    )
                    entry["has_gates"] = False
                    entries.append(entry)
                del statistics, empty_states

            with autocast_context(device, str(config.runtime.compute_dtype)):
                assistant_loss, auxiliary_loss, gradient = loss_and_hidden_gradient(
                    model,
                    updated_states,
                    ids,
                    assistant,
                    valid,
                    float(config.adaptation.aux_loss_weight),
                    int(config.runtime.loss_sequence_block_size),
                )
            torch.autograd.backward(updated_states, gradient)
            model.update_state()
            print(
                f"batch={batch_start // batch_size + 1} "
                f"episode={position:02d} "
                f"assistant_loss={assistant_loss.item():.5f} "
                f"aux_loss={auxiliary_loss.item():.5f}",
                flush=True,
            )
            del updated_states, gradient

        model.empty_state()
        del input_ids, assistant_mask, valid_mask, batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return entries


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="visualize_oloop_lora",
)
def main(config: DictConfig) -> None:
    if config.sampling.latents is not None:
        config.sampling.mode = "trajectories"
    if config.sampling.mode not in ("episodes", "trajectories"):
        raise ValueError("sampling.mode must be 'episodes' or 'trajectories'")
    device = torch.device(str(config.runtime.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda but CUDA is unavailable")
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.set_float32_matmul_precision("high")

    model_config = resolved_model_config(config)
    configured_output = Path(str(config.output.directory))
    output_base = (
        configured_output
        if configured_output.is_absolute()
        else constants.BASE_PATH / configured_output
    )
    run_output_root = (
        output_base
        / model_url_folder(model_config)
        / filename_stem(config.name)
    )
    run_output_root.mkdir(parents=True, exist_ok=True)
    data_root = run_output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(config), flush=True)
    trajectories, subsets = sample_trajectories(config)
    model, model_config = load_model(config, device)
    entries = collect(config, model, trajectories, run_output_root, device)
    outputs = write_outputs(config, run_output_root, run_output_root, entries)

    manifest = {
        "config": OmegaConf.to_container(config, resolve=True),
        "resolved_model_config": OmegaConf.to_container(model_config, resolve=True),
        "subsets": subsets,
        "trajectory_count": len(trajectories),
        "visualized_episode_count": len(entries),
        "data_directory": "data",
        "state_update_note": (
            "Each visualized episode uses OLoop-LoRA fast weights updated through "
            "every preceding episode. Base probabilities use empty fast weights. "
            "Adaptation gradients include the configured auxiliary loss."
        ),
        "entries": entries,
        "html_files": [
            str(path.relative_to(run_output_root)) for path in outputs
        ],
    }
    manifest_path = run_output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Generated:", flush=True)
    for path in outputs:
        print(path, flush=True)


if __name__ == "__main__":
    main()
