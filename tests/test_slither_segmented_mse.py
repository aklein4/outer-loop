import unittest

import torch
from omegaconf import OmegaConf

from models.slither import (
    SlitherStateMechanism,
    SlitherStateWriter,
    pcg_solve,
)


def _mechanism_config():
    return OmegaConf.create(
        {
            "hidden_size": 12,
            "state_size": 12,
            "num_state_in_heads": 3,
            "num_state_out_heads": 3,
            "rms_norm_eps": 1e-6,
            "pcg_iterations": 8,
            "init_mse_lambda": 0.05,
            "init_state_out_scale": 0.1,
        }
    )


class SegmentedMSETest(unittest.TestCase):

    def test_segmented_pcg_matches_direct_solve(self):
        torch.manual_seed(0)

        batch_size = 2
        num_segments = 3
        segment_size = 8
        num_queries = 5

        factors = torch.randn(
            batch_size,
            num_segments,
            segment_size,
            segment_size,
            dtype=torch.float32,
        )
        matrix = factors.mT @ factors
        matrix = matrix + 0.25 * torch.eye(segment_size)
        rhs = torch.randn(
            batch_size,
            num_segments,
            segment_size,
            num_queries,
            dtype=torch.float32,
        )

        actual = pcg_solve(
            matrix,
            rhs,
            # Exact arithmetic needs at most d iterations. FP32 loses some
            # conjugacy, so allow additional iterations for this direct-solve
            # equivalence test.
            iterations=2 * segment_size,
            eps=1e-6,
        )
        expected = torch.linalg.solve(matrix, rhs)

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_segmented_pcg_is_differentiable(self):
        torch.manual_seed(1)

        factor = torch.randn(1, 2, 6, 6)
        matrix = (factor.mT @ factor).requires_grad_()
        matrix = matrix + 0.5 * torch.eye(6)
        rhs = torch.randn(1, 2, 6, 3, requires_grad=True)

        solution = pcg_solve(
            matrix,
            rhs,
            iterations=6,
            eps=1e-6,
        )
        loss = solution.square().sum()
        matrix_grad, rhs_grad = torch.autograd.grad(loss, (matrix, rhs))

        self.assertTrue(torch.isfinite(matrix_grad).all())
        self.assertTrue(torch.isfinite(rhs_grad).all())
        self.assertGreater(matrix_grad.abs().sum(), 0)
        self.assertGreater(rhs_grad.abs().sum(), 0)

    def test_pcg_backward_matches_solve_on_low_rank_system(self):
        # Low-rank key correlations are common early in an episode. Unrolling
        # the finite-precision CG recurrence through these systems produces
        # gradients thousands of times larger than the linear-solve gradient,
        # even when the forward approximation is already accurate.
        torch.manual_seed(32101)

        dimension = 16
        keys = torch.randn(1, 2, 128, 1)
        mixing = torch.randn(1, 2, 1, dimension)
        features = keys @ mixing
        matrix = features.mT @ features / features.shape[-2]
        matrix = matrix + 0.1 * torch.eye(dimension)
        rhs = torch.randn(1, 2, dimension, 8)
        output_gradient = torch.randn(1, 8, 2 * dimension)

        def normalized_loss(solution):
            solution = solution.permute(0, 3, 1, 2).reshape(
                1, 8, 2 * dimension
            )
            solution = torch.nn.functional.rms_norm(
                solution, [2 * dimension], eps=1e-5
            )
            return (solution * output_gradient).mean()

        pcg_matrix = matrix.detach().requires_grad_()
        pcg_rhs = rhs.detach().requires_grad_()
        pcg_solution = pcg_solve(
            pcg_matrix, pcg_rhs, iterations=10, eps=1e-5
        )
        pcg_grads = torch.autograd.grad(
            normalized_loss(pcg_solution),
            (pcg_matrix, pcg_rhs),
        )

        exact_matrix = matrix.detach().requires_grad_()
        exact_rhs = rhs.detach().requires_grad_()
        exact_solution = torch.linalg.solve(exact_matrix, exact_rhs)
        exact_grads = torch.autograd.grad(
            normalized_loss(exact_solution),
            (exact_matrix, exact_rhs),
        )

        for pcg_grad, exact_grad in zip(pcg_grads, exact_grads):
            relative_error = (
                (pcg_grad - exact_grad).norm() / exact_grad.norm()
            )
            self.assertLess(relative_error, 5e-2)

    def test_pcg_zero_rhs_remains_finite(self):
        matrix = torch.eye(4).expand(2, 3, 4, 4)
        rhs = torch.zeros(2, 3, 4, 5)

        solution = pcg_solve(matrix, rhs, iterations=8, eps=1e-6)

        self.assertTrue(torch.isfinite(solution).all())
        self.assertTrue(torch.equal(solution, torch.zeros_like(solution)))

    def test_writer_accumulates_segmented_key_gram(self):
        torch.manual_seed(2)
        config = _mechanism_config()
        writer = SlitherStateWriter(config)
        memory = torch.randn(2, 7, config.hidden_size)

        update, gram, count = writer(memory)

        key = writer.activation(writer.k_proj(memory))
        key_gate = torch.softmax(writer.in_gate(memory), dim=-1)
        key_gate = key_gate * writer.num_state_in_heads
        key = writer.in_norm(key, scales=key_gate)
        key = key.view(2, 7, config.num_state_in_heads, 4)
        expected_gram = torch.einsum("blhd,blhe->bhde", key, key)

        self.assertEqual(
            update.shape,
            (2, config.state_size, config.state_size),
        )
        self.assertEqual(
            gram.shape,
            (2, config.num_state_in_heads, 4, 4),
        )
        torch.testing.assert_close(gram, expected_gram)
        torch.testing.assert_close(gram, gram.mT)
        self.assertTrue(
            torch.equal(count, torch.full((2,), 7, dtype=torch.int32))
        )

    def test_mechanism_solve_matches_segmented_normal_equations(self):
        torch.manual_seed(4)
        mechanism = SlitherStateMechanism(_mechanism_config())
        mechanism.init_state(bs=2, device=torch.device("cpu"))

        factors = torch.randn(2, 3, 4, 9)
        gram = factors @ factors.mT
        with torch.no_grad():
            mechanism.k_corr.copy_(gram)
            mechanism.k_count.fill_(9)

        query = torch.randn(2, 5, 12)
        actual = mechanism._solve(query)

        mean_gram = gram / 9
        regularizer = mechanism.get_lambda()
        matrix = (
            mean_gram
            + regularizer[None, :, :, None]
            * torch.eye(4)[None, None]
        )
        rhs = query.view(2, 5, 3, 4).permute(0, 2, 3, 1)
        expected = torch.linalg.solve(matrix, rhs)
        expected = expected.permute(0, 3, 1, 2).reshape_as(query)

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_lambda_is_diagonal_positive_and_correctly_initialized(self):
        mechanism = SlitherStateMechanism(_mechanism_config())

        regularizer = mechanism.get_lambda()

        self.assertEqual(regularizer.shape, (3, 4))
        self.assertTrue((regularizer > 0).all())
        torch.testing.assert_close(
            regularizer,
            torch.full_like(
                regularizer,
                _mechanism_config().init_mse_lambda
                + _mechanism_config().rms_norm_eps,
            ),
        )

    def test_state_update_is_reconstructed_by_decrement(self):
        torch.manual_seed(3)
        mechanism = SlitherStateMechanism(_mechanism_config())
        mechanism.init_state(bs=2, device=torch.device("cpu"))
        memory = torch.randn(2, 7, 12)

        mechanism.increment_state(memory)
        self.assertTrue(
            torch.equal(
                mechanism.k_count,
                torch.full((2,), 7, dtype=torch.int32),
            )
        )
        mechanism.decrement_state(memory)

        torch.testing.assert_close(
            mechanism.state,
            torch.zeros_like(mechanism.state),
            rtol=0,
            atol=2e-6,
        )
        torch.testing.assert_close(
            mechanism.k_corr,
            torch.zeros_like(mechanism.k_corr),
            rtol=0,
            atol=2e-6,
        )
        self.assertTrue(
            torch.equal(mechanism.k_count, torch.zeros_like(mechanism.k_count))
        )


if __name__ == "__main__":
    unittest.main()
