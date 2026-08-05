import sys
import unittest
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import utils.torch_utils as torch_utils
from models.forte import ForteMode, ForteModel


def eager_scan(function, carry, iterable_tensors, **_kwargs):
    outputs = []
    for values in zip(*iterable_tensors):
        carry, output = function(carry, values)
        outputs.append(output)
    return carry, torch.stack(outputs)


class ToyFastModel(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, width))

    def body(self, x, target, state, grad_buffer, with_gradients):
        state = state.detach().requires_grad_(True)
        grad_buffer = grad_buffer.detach().requires_grad_(True)
        prediction = (
            x @ self.weight
            + torch.einsum("bij,bj->bi", state, x)
            + 0.1 * torch.einsum("bij,bj->bi", grad_buffer, x)
        )
        loss = (prediction - target).square().sum()
        state_grad, buffer_grad, weight_grad = torch.autograd.grad(
            loss, (state, grad_buffer, self.weight),
        )
        return (
            loss.detach(),
            (weight_grad,) if with_gradients else None,
            (state - 0.05 * state_grad).detach(),
            (grad_buffer + buffer_grad).detach(),
        )


def python_loop(function, iterable_tensors, *carry):
    loss = iterable_tensors[0].new_zeros(())
    gradients = None
    for values in zip(*iterable_tensors):
        value, step_gradients, *carry = function(*values, *carry)
        loss = loss + value
        if step_gradients is not None:
            if gradients is None:
                gradients = [torch.zeros_like(g) for g in step_gradients]
            for total, gradient in zip(gradients, step_gradients):
                total.add_(gradient)
    return loss, gradients, carry


class ScannedTrainingLoopTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.original_scan = getattr(torch_utils, "xla_scan", None)
        torch_utils.xla_scan = eager_scan

    @classmethod
    def tearDownClass(cls):
        if cls.original_scan is None:
            del torch_utils.xla_scan
        else:
            torch_utils.xla_scan = cls.original_scan

    def inputs(self):
        torch.manual_seed(1234)
        episodes, batch, width = 5, 6, 4
        x = torch.randn(episodes, batch, width, device="cuda")
        target = torch.randn_like(x)
        state = torch.randn(batch, width, width, device="cuda")
        grad_buffer = torch.randn_like(state)
        return x, target, state, grad_buffer

    def test_first_pass_outputs_match_python_loop(self):
        x, target, state, grad_buffer = self.inputs()
        model = ToyFastModel(x.shape[-1]).cuda()
        function = lambda *args: model.body(*args, with_gradients=False)

        expected_loss, _, expected_carry = python_loop(
            function, (x, target), state, grad_buffer,
        )
        scan = torch_utils.ScannedTrainingLoop(
            model, function, accumulate_gradients=False,
        )
        loss, *carry = scan((x, target), state, grad_buffer)

        torch.testing.assert_close(loss, expected_loss)
        for actual, expected in zip(carry, expected_carry):
            torch.testing.assert_close(actual, expected)
        self.assertIsNone(model.weight.grad)

    def test_no_gradient_scan_accepts_integer_inputs(self):
        indices = torch.arange(5, device="cuda", dtype=torch.long)
        initial = torch.zeros((), device="cuda")

        def function(index, carry):
            carry = carry + index
            return carry, None, carry

        expected_loss, _, expected_carry = python_loop(
            function, (indices,), initial,
        )
        model = nn.Linear(1, 1).cuda()
        loss, carry = torch_utils.ScannedTrainingLoop(
            model, function, accumulate_gradients=False,
        )((indices,), initial)
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(carry, expected_carry[0])

    def test_second_pass_outputs_gradients_and_shards_match(self):
        x, target, state, grad_buffer = self.inputs()
        reference = ToyFastModel(x.shape[-1]).cuda()
        initial_weight = reference.weight.detach().clone()
        function = lambda *args: reference.body(*args, with_gradients=True)
        expected_loss, expected_gradients, expected_carry = python_loop(
            function, (x, target), state, grad_buffer,
        )

        scanned = ToyFastModel(x.shape[-1]).cuda()
        scanned.weight.data.copy_(initial_weight)
        scan = torch_utils.ScannedTrainingLoop(
            scanned,
            lambda *args: scanned.body(*args, with_gradients=True),
            accumulate_gradients=True,
        )
        loss, *carry = scan((x, target), state, grad_buffer)

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(scanned.weight.grad, expected_gradients[0])
        for actual, expected in zip(carry, expected_carry):
            torch.testing.assert_close(actual, expected)

        shard_gradients = []
        shard_carries = []
        shard_losses = []
        for shard in (slice(0, 3), slice(3, 6)):
            model = ToyFastModel(x.shape[-1]).cuda()
            model.weight.data.copy_(initial_weight)
            shard_scan = torch_utils.ScannedTrainingLoop(
                model,
                lambda *args, model=model: model.body(
                    *args, with_gradients=True,
                ),
                accumulate_gradients=True,
            )
            shard_loss, *shard_carry = shard_scan(
                (x[:, shard], target[:, shard]),
                state[shard],
                grad_buffer[shard],
            )
            shard_losses.append(shard_loss)
            shard_gradients.append(model.weight.grad)
            shard_carries.append(shard_carry)

        torch.testing.assert_close(sum(shard_losses), loss)
        torch.testing.assert_close(sum(shard_gradients), scanned.weight.grad)
        for index, expected in enumerate(carry):
            torch.testing.assert_close(
                torch.cat([part[index] for part in shard_carries]),
                expected,
            )

    def test_fast_weight_layer_scan_matches_original_update(self):
        torch.manual_seed(5678)
        layers, batch, width = 3, 4, 5
        modules = []
        for _ in range(layers):
            dynamic_lr = SimpleNamespace(
                log_lr=nn.Parameter(torch.randn(width, width, device="cuda")),
                scalar_scaler=math.sqrt(width),
                base_lr=0.2,
                fast_weight_size=width,
            )
            modules.append(SimpleNamespace(fast_dynamic_lr=dynamic_lr))
        model = SimpleNamespace(fast_modules=lambda: modules)

        tensors = [
            torch.randn(
                layers, batch, width, width,
                device="cuda", requires_grad=True,
            )
            for _ in range(4)
        ]
        states, buffers, updates, raw_gradients = tensors
        actual = ForteModel.functional_update_state(
            model,
            states,
            buffers,
            updates,
            raw_gradients,
            ForteMode.TRAIN_FIRST,
        )
        actual_loss = sum(value.square().sum() for value in actual)
        actual_gradients = torch.autograd.grad(
            actual_loss,
            (*tensors, *(m.fast_dynamic_lr.log_lr for m in modules)),
        )

        reference_tensors = [value.detach().requires_grad_(True) for value in tensors]
        reference_lrs = [
            module.fast_dynamic_lr.log_lr.detach().requires_grad_(True)
            for module in modules
        ]
        reference_states = []
        reference_buffers = []
        for index, log_lr in enumerate(reference_lrs):
            lr = torch.exp(
                log_lr[None] * math.sqrt(width)
                + math.log(0.2) - math.log(width)
            )
            reference_states.append(
                reference_tensors[0][index]
                - lr * reference_tensors[2][index]
            )
            reference_buffers.append(
                reference_tensors[1][index] + reference_tensors[3][index]
            )
        expected = (
            torch.stack(reference_states),
            torch.stack(reference_buffers),
        )
        expected_loss = sum(value.square().sum() for value in expected)
        expected_gradients = torch.autograd.grad(
            expected_loss,
            (*reference_tensors, *reference_lrs),
        )

        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value)
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients,
        ):
            torch.testing.assert_close(actual_gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
