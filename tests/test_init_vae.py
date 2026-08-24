import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from scripts.init_vae import isotropic_output_weight


def test_isotropic_output_weight_matches_target_spectrum_and_scale():
    torch.manual_seed(0)
    target_basis, _ = torch.linalg.qr(torch.randn(7, 7))
    target_values = torch.linspace(0.25, 3.0, 7)
    target_covariance = (
        target_basis @ torch.diag(target_values) @ target_basis.T
    )

    scale = 0.1
    weight = isotropic_output_weight(
        target_covariance,
        input_size=5,
        scale=scale,
    )
    output_values = torch.linalg.eigvalsh(weight @ weight.T)

    expected = torch.cat(
        (torch.zeros(2), target_values[-5:] * scale**2)
    )
    torch.testing.assert_close(output_values, expected, atol=1e-6, rtol=1e-5)


def test_isotropic_output_weight_rejects_negative_scale():
    covariance = torch.eye(2)
    try:
        isotropic_output_weight(covariance, input_size=2, scale=-0.1)
    except ValueError as error:
        assert "latent_mlp_init_scale" in str(error)
    else:
        raise AssertionError("expected a negative scale to be rejected")
