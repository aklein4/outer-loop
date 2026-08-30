import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from models.forte import ForteMode, ForteModel, _get_G
from scripts.initialize_forte import (
    initialize_controller_input,
    initialize_fast_input,
)


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
            "grad_rms_eps": 1e-12,
            "mixer_kernel_size": 3,
            "pad_attention_bias_value": -1000.0,
        }
    )


def test_get_G_returns_learning_rate_scaled_update():
    torch.manual_seed(0)
    activations = torch.randn(2, 4, 3)
    output_grad = torch.randn(2, 4, 5)
    down_weight = torch.randn(5, 3)
    activation_gate_logits = torch.randn(2, 4, 3)
    gradient_gate_logits = torch.randn(2, 4, 3)
    token_gate_logits = torch.randn(2, 4, 1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    lr = torch.rand(1, 3, 3)

    raw_G, update = _get_G(
        activations,
        output_grad,
        down_weight,
        activation_gate_logits,
        gradient_gate_logits,
        token_gate_logits,
        mask,
        1e-12,
        lr,
    )
    doubled_G, doubled_update = _get_G(
        activations,
        output_grad,
        down_weight,
        activation_gate_logits,
        gradient_gate_logits,
        token_gate_logits,
        mask,
        1e-12,
        2 * lr,
    )

    torch.testing.assert_close(doubled_G, raw_G)
    torch.testing.assert_close(doubled_update, 2 * update)


def test_piano_initialization_conventions_are_applied_to_forte_components():
    torch.manual_seed(2)
    model = ForteModel(tiny_config())
    mlp = model.fast_modules()[0]
    inputs = torch.randn(3, 7, model.config.hidden_size) + 0.5
    embeddings = torch.randn_like(inputs) - 0.25
    mask = torch.tensor(
        [[1] * 7, [1] * 5 + [0] * 2, [1] * 4 + [0] * 3],
        dtype=torch.bool,
    )

    initialize_fast_input(mlp, (inputs,), mask, 0.25)
    initialize_controller_input(
        mlp.fast_dynamic_lr,
        embeddings,
        mask,
        0.25,
    )

    assert mlp.up_fast.bias is None
    projections_and_inputs = (
        (mlp.gate_fast, inputs),
        (mlp.fast_dynamic_lr.activation_gate_proj, embeddings),
        (mlp.fast_dynamic_lr.gradient_gate_proj, embeddings),
        (mlp.fast_dynamic_lr.token_gate_proj, embeddings),
    )
    for projection, values in projections_and_inputs:
        assert projection.bias is not None
        projected = projection(values)[mask]
        torch.testing.assert_close(
            projected.mean(dim=0),
            torch.zeros(projected.shape[-1]),
            atol=2e-5,
            rtol=2e-5,
        )


def test_double_pass_propagates_controller_grads_and_manages_scaled_state():
    torch.manual_seed(1)
    model = ForteModel(tiny_config())
    batch_size, sequence_length = 2, 5
    input_ids = torch.randint(0, model.vocab_size, (batch_size, sequence_length))
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    model.init_state(batch_size, torch.device("cpu"))

    with torch.no_grad():
        inferred = model.forward_backbone(input_ids, mode=ForteMode.INFERENCE)
        embeddings = model.forward_embeddings(inferred, mask).detach()
    hidden = model.forward_backbone(
        input_ids, embeddings, mask, mode=ForteMode.TRAIN_FIRST
    )
    states = model.forward_lm_states(
        hidden, embeddings, mask, mode=ForteMode.TRAIN_FIRST
    )
    torch.autograd.backward(
        states,
        torch.randn_like(states),
        inputs=model.grad_containers(),
    )

    expected_updates = [mlp.state.grad.detach().clone() for mlp in model.fast_modules()]
    model.update_state(ForteMode.TRAIN_FIRST)
    for mlp, expected in zip(model.fast_modules(), expected_updates):
        torch.testing.assert_close(mlp.state, expected)

    model.finalize_state()
    model.zero_grad(set_to_none=False)

    doubled_ids = torch.repeat_interleave(input_ids, 2, dim=0)
    inferred = model.forward_backbone(input_ids, mode=ForteMode.INFERENCE)
    embeddings = model.forward_embeddings(inferred, mask)
    doubled_embeddings = torch.repeat_interleave(embeddings, 2, dim=0)
    hidden = model.forward_backbone(
        doubled_ids,
        doubled_embeddings,
        mask,
        mode=ForteMode.TRAIN_SECOND,
    )
    states = model.forward_lm_states(
        hidden,
        doubled_embeddings,
        mask,
        mode=ForteMode.TRAIN_SECOND,
    )[::2]
    states.backward(torch.randn_like(states))

    assert model.embedding_state_shift.grad is not None
    assert model.embedding_state_shift.grad.abs().sum() > 0
    assert model.embed_tokens.weight.grad is not None
    assert model.embed_tokens.weight.grad.abs().sum() > 0
    assert all(mlp.state.grad.abs().sum() > 0 for mlp in model.fast_modules())

    model.update_state(ForteMode.TRAIN_SECOND)
