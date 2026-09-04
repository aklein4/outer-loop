import sys
from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from models.forte import ForteMode, ForteModel
from scripts.initialize_forte import initialize_embedding_state


def tiny_config():
    return OmegaConf.create(
        {
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_output_layers": 1,
            "num_bidirectional_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "silu",
            "max_position_embeddings": 64,
            "rope_theta": 10_000.0,
            "rope_scaling": None,
            "initializer_range": 0.02,
            "attention_dropout": False,
            "attention_bias": False,
            "rms_norm_eps": 1e-5,
            "attention_kernel": None,
            "fast_weight_size": 8,
            "base_lr": 1e-3,
            "offset_alpha": 0.25,
            "num_fast_weight_heads": 2,
            "grad_rms_eps": 1e-12,
            "mixer_kernel_size": 3,
            "pad_attention_bias_value": -1000.0,
        }
    )


def test_forte_matches_piano_shared_hyperparameters_and_sharding():
    piano_model = OmegaConf.load(
        SRC / "configs/model/piano-llama3p2-1b.yaml"
    )
    forte_model = OmegaConf.load(
        SRC / "configs/model/forte-llama3p2-1b.yaml"
    )
    for name in ("fast_weight_size", "base_lr", "grad_rms_eps"):
        assert forte_model[name] == piano_model[name]

    piano_trainer = OmegaConf.to_container(
        OmegaConf.load(SRC / "configs/trainer/piano-xl.yaml"),
        resolve=True,
    )
    forte_trainer = OmegaConf.to_container(
        OmegaConf.load(SRC / "configs/trainer/forte-xl.yaml"),
        resolve=True,
    )
    piano_trainer.pop("type")
    forte_trainer.pop("type")
    assert forte_trainer == piano_trainer

    piano_sharding = OmegaConf.load(
        SRC / "configs/model/sharding/piano-fsdp.yaml"
    )
    forte_sharding = OmegaConf.load(
        SRC / "configs/model/sharding/forte-fsdp.yaml"
    )
    assert forte_sharding.replicate_default == piano_sharding.replicate_default
    weight_spec = ["fsdp", None]
    for stack in (
        "backbone_layers.layers.*.mlp",
        "output_layers.layers.*.mlp",
        "bidirectional_head.layers.layers.*.mlp",
    ):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert forte_sharding[f"{stack}.{projection}.weight"] == weight_spec

    activation_spec = [["data", "fsdp"], None, None]
    for stack in (
        "backbone_layers.layers.*",
        "output_layers.layers.*",
        "bidirectional_head.layers.layers.*",
    ):
        assert forte_sharding[stack] == activation_spec


def first_pass(model, input_ids, mask, output_gradient):
    with torch.no_grad():
        hidden = model.forward_backbone(input_ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(hidden, mask).detach()
    hidden = model.forward_backbone(
        input_ids, embeddings, mask, mode=ForteMode.TRAIN_FIRST
    )
    states = model.forward_lm_states(
        hidden, embeddings, mask, mode=ForteMode.TRAIN_FIRST
    )
    torch.autograd.backward(
        states,
        output_gradient,
        inputs=model.grad_containers(),
    )
    return states


def second_pass(model, input_ids, mask, output_gradient):
    doubled_ids = torch.repeat_interleave(input_ids, 2, dim=0)
    hidden = model.forward_backbone(input_ids, mode=ForteMode.INFERENCE)
    embeddings = model.forward_embeddings(hidden, mask)
    hidden = model.forward_backbone(
        doubled_ids, embeddings, mask, mode=ForteMode.TRAIN_SECOND
    )
    states = model.forward_lm_states(
        hidden, embeddings, mask, mode=ForteMode.TRAIN_SECOND
    )[::2]
    states.backward(output_gradient)
    return states


def test_scaled_state_update_and_single_controller_double_causal_pass():
    torch.manual_seed(1)
    model = ForteModel(tiny_config())
    input_ids = torch.randint(0, model.vocab_size, (2, 5))
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    model.init_state(input_ids.shape[0], torch.device("cpu"))

    output_gradient = torch.randn(2, 5, model.config.hidden_size)
    first_pass(model, input_ids, mask, output_gradient)
    expected_updates = [mlp.state.grad.detach().clone() for mlp in model.fast_modules()]
    model.update_state(ForteMode.TRAIN_FIRST)
    for mlp, expected in zip(model.fast_modules(), expected_updates):
        torch.testing.assert_close(mlp.state, expected)

    model.finalize_state()
    model.zero_grad(set_to_none=False)
    second_pass(model, input_ids, mask, torch.randn_like(output_gradient))

    assert model.embedding_state_shift.grad is not None
    assert model.embedding_state_shift.grad.abs().sum() > 0
    assert model.embed_tokens.weight.grad is not None
    assert model.embed_tokens.weight.grad.abs().sum() > 0


def test_identical_passes_cancel_raw_gradient_buffer():
    torch.manual_seed(2)
    model = ForteModel(tiny_config())
    input_ids = torch.randint(0, model.vocab_size, (2, 5))
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    model.init_state(input_ids.shape[0], torch.device("cpu"))

    output_gradient = torch.randn(2, 5, model.config.hidden_size)
    first_states = first_pass(model, input_ids, mask, output_gradient)
    model.update_state(ForteMode.TRAIN_FIRST)
    model.finalize_state()
    model.zero_grad(set_to_none=False)

    second_states = second_pass(model, input_ids, mask, output_gradient)
    torch.testing.assert_close(
        second_states,
        first_states.detach(),
        atol=1e-6,
        rtol=1e-5,
    )
    model.update_state(ForteMode.TRAIN_SECOND)

    max_error = max(
        float(mlp.relative_grad_error().max())
        for mlp in model.fast_modules()
    )
    assert max_error < 1e-4


def test_embedding_calibration_hook_is_exercised():
    torch.manual_seed(3)
    model = ForteModel(tiny_config())
    hidden_states = torch.randn(3, 7, model.config.hidden_size)
    mask = torch.tensor(
        [[1] * 7, [1] * 5 + [0] * 2, [1] * 4 + [0] * 3],
        dtype=torch.bool,
    )
    handle = model.embedding_norm.register_forward_hook(
        partial(initialize_embedding_state, model, mask=mask)
    )
    try:
        model.forward_embeddings(hidden_states, mask)
    finally:
        handle.remove()

    assert model.embedding_state_shift.abs().sum() > 0
    assert model.embedding_state_scale.abs().sum() > 0


@pytest.mark.parametrize(
    "evaluator_name",
    ["evaluate_icl", "evaluate_persona", "evaluate_policy"],
)
def test_evaluator_uses_forte_first_pass_and_inference_modes(evaluator_name):
    torch.manual_seed(4)
    evaluator = import_module(evaluator_name)
    model = ForteModel(tiny_config())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.embed_tokens.requires_grad_(True)

    device = torch.device("cpu")
    batch_size, sequence_length = 2, 5
    model.init_state(batch_size, device)
    args = SimpleNamespace(dtype="float32", aux_weight=0.0, compile=False)
    train_fn, inference_fn = evaluator.make_fns(model, args, device)

    input_ids = torch.randint(0, model.vocab_size, (batch_size, sequence_length))
    assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    assistant_mask[:, -2:] = True
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    old_states = [state.detach().clone() for state in model.state_containers()]

    train_fn(
        input_ids,
        assistant_mask,
        attention_mask,
        torch.ones(1, 1, 1),
    )

    assert any(
        not torch.equal(old_state, new_state)
        for old_state, new_state in zip(old_states, model.state_containers())
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    if evaluator_name == "evaluate_policy":
        prompt_indices = torch.full((batch_size,), sequence_length - 1)
        token_ids = torch.arange(4)
        assert inference_fn(
            input_ids,
            attention_mask,
            prompt_indices,
            token_ids,
        ).shape == (batch_size, 4)
    else:
        assert inference_fn(input_ids).shape == (
            batch_size,
            sequence_length - 1,
            model.vocab_size,
        )
