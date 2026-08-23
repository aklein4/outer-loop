import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from models.llama import LlamaForCausalLM
from models.vae import VAEModel
from utils.torch_utils import set_no_muon


def tiny_config():
    with initialize_config_dir(version_base=None, config_dir=str(SRC / "configs")):
        return compose(
            config_name="default",
            overrides=["model=vae-llama3p2-test", "trainer=vae-test"],
        ).model


def test_vae_forward_normalizes_mu_and_uses_standard_autograd():
    torch.manual_seed(0)
    config = tiny_config()
    config.vocab_size = 64
    model = VAEModel(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 7))
    valid_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    alpha = torch.tensor(0.2)
    noise = torch.zeros(2, config.latent_size, config.latent_size)
    radius = torch.tensor([0.25, 0.75])
    logits, mu, sampled_radius = model(
        input_ids,
        valid_mask,
        alpha=alpha,
        noise=noise,
        radius=radius,
        logits_to_keep=slice(0, -1),
    )

    assert logits.shape == (2, 6, config.vocab_size)
    assert mu.shape == (2, config.latent_size, config.latent_size)
    torch.testing.assert_close(sampled_radius, radius)
    torch.testing.assert_close(
        mu.float().square().mean(dim=(-2, -1)).sqrt(),
        torch.full((2,), 0.2),
        atol=2e-5,
        rtol=2e-5,
    )

    logits.square().mean().backward()
    assert model.latent_writer.key_proj.weight.grad is not None
    assert all(module.odot.grad is not None for module in model.latent_modules())


def test_radius_scaled_sampling_and_decoder_frobenius_normalization():
    config = tiny_config()
    model = VAEModel(config)
    mu = torch.randn(2, config.latent_size, config.latent_size)
    noise = torch.randn_like(mu)
    radius = torch.tensor([0.2, 0.8])
    sampled = model.sample_latent(
        mu,
        noise_scale=0.5,
        noise=noise,
        radius=radius,
    )
    torch.testing.assert_close(
        sampled,
        radius[:, None, None] * mu + 0.5 * noise,
    )
    normalized = model.frobenius_rms_norm(
        sampled, config.latent_rms_norm_eps
    )
    torch.testing.assert_close(
        normalized.square().mean(dim=(-2, -1)).sqrt(),
        torch.ones(2),
        atol=3e-5,
        rtol=3e-5,
    )


def test_only_decoder_inputs_receive_radius_conditioning():
    torch.manual_seed(3)
    config = tiny_config()
    config.vocab_size = 64
    model = VAEModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    latent = torch.randn(2, config.latent_size, config.latent_size)

    captured = []
    handle = model.decoder_layers.layers[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        model.decode(input_ids, latent, radius=torch.zeros(2))
        model.decode(input_ids, latent, radius=torch.ones(2))
    finally:
        handle.remove()

    expected = model.decoder_radius_embedding[None, None].expand_as(captured[0])
    torch.testing.assert_close(captured[1] - captured[0], expected)


def test_encoder_scale_and_shift_precede_bidirectional_layers():
    torch.manual_seed(5)
    config = tiny_config()
    config.vocab_size = 64
    model = VAEModel(config)
    model.encoder_state_shift.data.copy_(
        torch.linspace(-0.2, 0.2, config.hidden_size)
    )
    model.encoder_state_scale.data.copy_(
        torch.linspace(-0.1, 0.1, config.hidden_size)
    )
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    mask = torch.ones_like(input_ids, dtype=torch.bool)

    normalized = []
    conditioned = []
    norm_handle = model.encoder_bidirectional_norm.register_forward_hook(
        lambda _module, _inputs, output: normalized.append(output.detach().clone())
    )
    head_handle = model.encoder_noncausal.register_forward_pre_hook(
        lambda _module, inputs: conditioned.append(inputs[0].detach().clone())
    )
    try:
        model.encode(input_ids, mask, alpha=torch.tensor(0.1))
    finally:
        norm_handle.remove()
        head_handle.remove()

    expected = (
        normalized[0] + model.encoder_state_shift
    ) * (1.0 + model.encoder_state_scale)
    torch.testing.assert_close(conditioned[0], expected)


def test_masked_encoder_tokens_do_not_change_ridge_latent():
    torch.manual_seed(1)
    config = tiny_config()
    model = VAEModel(config)
    hidden = torch.randn(2, 5, config.hidden_size)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool
    )
    changed = hidden.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000

    actual = model.latent_writer(hidden, mask)
    expected = model.latent_writer(changed, mask)
    torch.testing.assert_close(actual, expected)


def test_ridge_correlation_uses_post_gate_value_magnitudes():
    torch.manual_seed(4)
    config = tiny_config()
    writer = VAEModel(config).latent_writer
    hidden = torch.randn(2, 5, config.hidden_size)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool
    )

    actual = writer(hidden, mask)
    keys = writer.key_norm(writer.key_proj(hidden))
    values = writer.value_norm(writer.value_proj(hidden))
    gate = 2.0 * torch.sigmoid(writer.value_gate_proj(hidden).float())
    assert torch.all((gate > 0.0) & (gate < 2.0))

    values_h = values.reshape(
        2, 5, writer.num_value_heads, writer.value_head_size
    ) * gate[..., None]
    gated_values = values_h.reshape(2, 5, writer.latent_size)
    magnitudes = gated_values.float().square().mean(dim=-1).sqrt()
    float_mask = mask[..., None].float()
    count = float_mask.sum(dim=1).clamp_min(1.0)
    masked_keys = keys.float() * float_mask
    masked_values = gated_values.float() * float_mask
    cross = torch.einsum(
        "bso,bsi->boi", masked_values, masked_keys
    ) / count[:, None]
    keys_h = masked_keys.reshape(
        2, 5, writer.num_key_heads, writer.key_head_size
    )
    corr = torch.einsum(
        "bshd,bshe->bhde",
        keys_h * magnitudes[..., None, None],
        keys_h,
    ) / count[:, None, None]
    matrix = corr + torch.diag_embed(writer.get_lambda().float())[None]
    rhs = cross.reshape(
        2, writer.latent_size, writer.num_key_heads, writer.key_head_size
    ).permute(0, 2, 3, 1)
    expected = torch.linalg.solve(matrix, rhs).permute(0, 3, 1, 2).reshape(
        2, writer.latent_size, writer.latent_size
    )
    torch.testing.assert_close(actual, expected)

    assert writer.num_key_heads != writer.num_value_heads


def test_plain_llama_checkpoint_is_replicated_into_vae_stacks():
    torch.manual_seed(2)
    config = tiny_config()
    config.vocab_size = 64
    llama = LlamaForCausalLM(config)
    llama.model.norm.weight.data.copy_(
        torch.linspace(0.5, 1.5, config.hidden_size)
    )
    vae = VAEModel(config)
    vae.load_state_dict(llama.state_dict(), strict=False)

    torch.testing.assert_close(
        vae.embed_tokens.weight, llama.model.embed_tokens.weight
    )
    torch.testing.assert_close(
        vae.encoder_norm.weight, llama.model.norm.weight
    )
    torch.testing.assert_close(
        vae.decoder_norm.weight, llama.model.norm.weight
    )
    torch.testing.assert_close(
        vae.encoder_noncausal.norm.weight,
        torch.ones_like(vae.encoder_noncausal.norm.weight),
    )
    torch.testing.assert_close(
        vae.encoder_causal_layers.layers[0].self_attn.q_proj.weight,
        llama.model.layers.layers[0].self_attn.q_proj.weight,
    )
    assert len(vae.decoder_layers) == config.num_hidden_layers
    torch.testing.assert_close(
        vae.decoder_layers.layers[0].self_attn.q_proj.weight,
        llama.model.layers.layers[0].self_attn.q_proj.weight,
    )
    torch.testing.assert_close(
        vae.decoder_layers.layers[-1].self_attn.q_proj.weight,
        llama.model.layers.layers[-1].self_attn.q_proj.weight,
    )
    for index in range(config.num_encoder_causal_layers):
        torch.testing.assert_close(
            vae.encoder_causal_layers.layers[index].self_attn.q_proj.weight,
            llama.model.layers.layers[index].self_attn.q_proj.weight,
        )


def test_checkpoint_depth_is_enforced_during_initialization():
    config = tiny_config()
    config.vocab_size = 64
    llama = LlamaForCausalLM(config)
    shallow_state = {
        name: value
        for name, value in llama.state_dict().items()
        if not name.startswith("model.layers.layers.2.")
        and not name.startswith("model.layers.layers.3.")
        and not name.startswith("model.layers.layers.4.")
    }

    try:
        VAEModel(config).load_state_dict(shallow_state, strict=False)
    except ValueError as error:
        assert "num_encoder_causal_layers" in str(error)
        assert "checkpoint depth (2)" in str(error)
    else:
        raise AssertionError("expected shallow pretrained checkpoint to be rejected")


def test_shared_no_muon_patterns_cover_embeddings_and_latent_scalars():
    model = set_no_muon(VAEModel(tiny_config()))
    assert model.embed_tokens.weight.no_muon
    assert model.lm_head.weight.no_muon
    assert model.latent_writer.log_lambda.no_muon
    assert model.latent_writer.value_gate_proj.weight.no_muon
    assert all(module.odot.no_muon for module in model.latent_modules())
    assert all(
        layer.mlp.latent_mlp is latent
        for layer, latent in zip(
            model.decoder_layers._iter_layers(), model.latent_modules()
        )
    )
