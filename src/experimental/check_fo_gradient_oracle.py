"""Compare the FO custom backward with an explicit two-example unroll.

This is intentionally a tiny, dense model.  With a horizon of two there is
only one fast-weight update, so the FO decomposition should be equal to the
ordinary autograd gradient (up to the BF16 projection used by the production
raw-gradient helper).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fo_ittt import FastWeightFunction


DEVICE = torch.device("cuda")
DTYPE = torch.float32
GRAD_EPS = 1.0e-12


class ToyModel(nn.Module):
    def __init__(self, hidden_size: int = 8, fast_size: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.fast_size = fast_size
        self.slow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.base = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_fast = nn.Linear(hidden_size, fast_size, bias=False)
        self.gate_fast = nn.Linear(hidden_size, fast_size, bias=False)
        self.down_fast = nn.Linear(fast_size, hidden_size, bias=False)
        self.embedding = nn.Linear(hidden_size, 1, bias=False)
        self.log_lr = nn.Parameter(torch.tensor(-4.0))

    def components(self, x, state):
        hidden = torch.tanh(self.slow(x))
        base_output = self.base(hidden)
        fast_hidden = self.up_fast(hidden) * F.silu(
            self.gate_fast(hidden)
        )
        if fast_hidden.shape[0] == state.shape[0]:
            fast_values = torch.einsum(
                "boi,bsi->bso", state, fast_hidden
            )
        elif fast_hidden.shape[0] == 2 * state.shape[0]:
            streams = fast_hidden.reshape(
                state.shape[0], 2, *fast_hidden.shape[1:]
            )
            fast_values = torch.einsum(
                "boi,bnsi->bnso", state, streams
            ).flatten(0, 1)
        else:
            raise ValueError("input batch does not match fast state")
        fast_output = self.down_fast(fast_values)
        return hidden, fast_hidden, base_output, fast_output

    def plain_forward(self, x, state):
        _, _, base_output, fast_output = self.components(x, state)
        return base_output + fast_output

    def custom_forward(
        self,
        x,
        state,
        grad_buffer,
        remaining_gradient=None,
        learning_rate=None,
    ):
        _, fast_hidden, base_output, fast_output = self.components(x, state)
        fast_output = FastWeightFunction.apply(
            fast_hidden,
            fast_output,
            self.down_fast.weight,
            grad_buffer,
            remaining_gradient,
            learning_rate,
            GRAD_EPS,
        )
        return base_output + fast_output

    def learning_rate(self, x, state):
        output = self.plain_forward(x, state)
        scalar = (
            self.log_lr
            + 0.2 * self.embedding(output).mean(dim=1).squeeze(-1)
        )
        return scalar.exp()[:, None, None].expand(
            -1, self.fast_size, self.fast_size
        )


class ToyBlock(nn.Module):
    def __init__(self, hidden_size, fast_size):
        super().__init__()
        self.slow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.base = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_fast = nn.Linear(hidden_size, fast_size, bias=False)
        self.gate_fast = nn.Linear(hidden_size, fast_size, bias=False)
        self.down_fast = nn.Linear(fast_size, hidden_size, bias=False)

    def forward(
        self,
        x,
        state,
        grad_buffer=None,
        remaining_gradient=None,
        learning_rate=None,
    ):
        hidden = torch.tanh(self.slow(x))
        base_output = self.base(hidden)
        fast_hidden = self.up_fast(hidden) * F.silu(
            self.gate_fast(hidden)
        )
        if fast_hidden.shape[0] == state.shape[0]:
            fast_values = torch.einsum(
                "boi,bsi->bso", state, fast_hidden
            )
        else:
            streams = fast_hidden.reshape(
                state.shape[0], 2, *fast_hidden.shape[1:]
            )
            fast_values = torch.einsum(
                "boi,bnsi->bnso", state, streams
            ).flatten(0, 1)
        fast_output = self.down_fast(fast_values)
        if grad_buffer is not None:
            fast_output = FastWeightFunction.apply(
                fast_hidden,
                fast_output,
                self.down_fast.weight,
                grad_buffer,
                remaining_gradient,
                learning_rate,
                GRAD_EPS,
            )
        return x + base_output + fast_output


class StackedToyModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 8,
        fast_size: int = 6,
        layer_count: int = 3,
        log_lr: float = -4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fast_size = fast_size
        self.layer_count = layer_count
        self.blocks = nn.ModuleList(
            [
                ToyBlock(hidden_size, fast_size)
                for _ in range(layer_count)
            ]
        )
        self.embedding = nn.Linear(hidden_size, layer_count, bias=False)
        self.log_lr = nn.Parameter(
            torch.full((layer_count,), log_lr)
        )

    def forward(
        self,
        x,
        state,
        grad_buffer=None,
        remaining_gradient=None,
        learning_rate=None,
    ):
        for layer, block in enumerate(self.blocks):
            x = block(
                x,
                state[layer],
                (
                    None
                    if grad_buffer is None
                    else grad_buffer[layer]
                ),
                (
                    None
                    if remaining_gradient is None
                    else remaining_gradient[layer]
                ),
                (
                    None
                    if learning_rate is None
                    else learning_rate[layer]
                ),
            )
        return x

    def learning_rate(self, x, state):
        output = self.forward(x, state)
        embedding = self.embedding(output).mean(dim=1)
        scalar = self.log_lr[None] + 0.2 * embedding
        return scalar.exp().transpose(0, 1)[
            :, :, None, None
        ].expand(-1, -1, self.fast_size, self.fast_size)


def loss_fn(output, target):
    return (output - target).square().mean()


def normalized_update(raw_gradient, learning_rate):
    return -learning_rate * F.rms_norm(
        raw_gradient.float(),
        raw_gradient.shape[-2:],
        eps=GRAD_EPS,
    )


def named_gradients(model):
    return {
        name: (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for name, parameter in model.named_parameters()
    }


def explicit_unroll(model, inputs, targets):
    batch_size = inputs[0].shape[0]
    state_shape = (
        (
            model.layer_count,
            batch_size,
            model.fast_size,
            model.fast_size,
        )
        if isinstance(model, StackedToyModel)
        else (
            batch_size,
            model.fast_size,
            model.fast_size,
        )
    )
    state = torch.zeros(
        *state_shape,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    output_0 = (
        model(inputs[0], state)
        if isinstance(model, StackedToyModel)
        else model.plain_forward(inputs[0], state)
    )
    loss_0 = loss_fn(output_0, targets[0])
    raw_0 = torch.autograd.grad(
        loss_0, state, create_graph=True
    )[0]
    learning_rate_0 = model.learning_rate(inputs[0], state)
    state_1 = state + normalized_update(raw_0, learning_rate_0)

    output_1 = (
        model(inputs[1], state_1)
        if isinstance(model, StackedToyModel)
        else model.plain_forward(inputs[1], state_1)
    )
    loss_1 = loss_fn(output_1, targets[1])
    (loss_0 + loss_1).backward()
    return named_gradients(model), raw_0.detach(), state_1.detach()


def detached_state_direct(model, inputs, targets, state_1):
    batch_size = inputs[0].shape[0]
    state_0 = torch.zeros_like(state_1)
    loss = (
        loss_fn(
            (
                model(inputs[0], state_0)
                if isinstance(model, StackedToyModel)
                else model.plain_forward(inputs[0], state_0)
            ),
            targets[0],
        )
        + loss_fn(
            (
                model(inputs[1], state_1.detach())
                if isinstance(model, StackedToyModel)
                else model.plain_forward(inputs[1], state_1.detach())
            ),
            targets[1],
        )
    )
    loss.backward()
    return named_gradients(model)


def custom_replay(model, inputs, targets):
    batch_size = inputs[0].shape[0]
    state_shape = (
        (
            model.layer_count,
            batch_size,
            model.fast_size,
            model.fast_size,
        )
        if isinstance(model, StackedToyModel)
        else (
            batch_size,
            model.fast_size,
            model.fast_size,
        )
    )
    state = torch.zeros(
        *state_shape,
        device=DEVICE,
        dtype=DTYPE,
    )
    countdown = torch.zeros_like(state, requires_grad=True)
    countdown.grad = torch.zeros_like(countdown)

    # First pass: collect raw gradients and construct the replay state.
    output_0 = (
        model(inputs[0], state, countdown)
        if isinstance(model, StackedToyModel)
        else model.custom_forward(inputs[0], state, countdown)
    )
    loss_0 = loss_fn(output_0, targets[0])
    torch.autograd.backward(loss_0, inputs=(countdown,))
    raw_0 = countdown.grad.detach().clone()
    with torch.no_grad():
        lr_0 = model.learning_rate(inputs[0], state)
        state.add_(normalized_update(raw_0, lr_0))
        countdown.add_(raw_0)
        countdown.grad.zero_()

    output_1 = (
        model(inputs[1], state, countdown)
        if isinstance(model, StackedToyModel)
        else model.custom_forward(inputs[1], state, countdown)
    )
    loss_1 = loss_fn(output_1, targets[1])
    torch.autograd.backward(loss_1, inputs=(countdown,))
    raw_1 = countdown.grad.detach().clone()
    with torch.no_grad():
        countdown.add_(raw_1)
        countdown.grad.zero_()
        state.zero_()

    model.zero_grad(set_to_none=True)

    # Second pass, episode zero: direct loss plus the future-loss correction.
    propagated_lr = model.learning_rate(inputs[0], state)
    lr_leaf = propagated_lr.detach().requires_grad_(True)
    double_input = inputs[0][:, None].expand(
        -1, 2, -1, -1
    ).flatten(0, 1)
    output = (
        model(
            double_input,
            state,
            countdown,
            countdown.detach(),
            lr_leaf,
        )
        if isinstance(model, StackedToyModel)
        else model.custom_forward(
            double_input,
            state,
            countdown,
            countdown.detach(),
            lr_leaf,
        )
    ).reshape(batch_size, 2, *targets[0].shape[1:])[:, 0]
    loss_fn(output, targets[0]).backward()
    (
        propagated_lr
        * lr_leaf.grad.detach().to(propagated_lr.dtype)
    ).sum().backward()
    replay_raw_0 = countdown.grad.detach().clone()
    with torch.no_grad():
        state.add_(normalized_update(replay_raw_0, lr_leaf))
        countdown.sub_(replay_raw_0)
        countdown.grad.zero_()

    # Terminal direct loss.
    output = (
        model(inputs[1], state, countdown)
        if isinstance(model, StackedToyModel)
        else model.custom_forward(inputs[1], state, countdown)
    )
    loss_fn(output, targets[1]).backward()

    return (
        named_gradients(model),
        raw_0,
        replay_raw_0,
        raw_1,
        countdown.detach().clone(),
        state.detach().clone(),
    )


def comparison(reference, actual):
    rows = []
    for name in reference:
        expected = reference[name].float()
        observed = actual[name].float()
        difference = observed - expected
        expected_norm = expected.norm()
        observed_norm = observed.norm()
        rows.append(
            (
                name,
                expected_norm.item(),
                observed_norm.item(),
                (
                    difference.norm()
                    / expected_norm.clamp_min(1.0e-30)
                ).item(),
                F.cosine_similarity(
                    expected.flatten(),
                    observed.flatten(),
                    dim=0,
                    eps=1.0e-30,
                ).item(),
            )
        )
    return rows


def subtract_gradients(left, right):
    return {name: left[name] - right[name] for name in left}


def run_case(model, label):
    torch.manual_seed(1234)
    model = model.to(device=DEVICE, dtype=DTYPE)
    replay_model = copy.deepcopy(model)
    direct_model = copy.deepcopy(model)
    inputs = [
        torch.randn(2, 5, model.hidden_size, device=DEVICE)
        for _ in range(2)
    ]
    targets = [
        torch.randn(2, 5, model.hidden_size, device=DEVICE)
        for _ in range(2)
    ]

    expected, exact_raw_0, exact_state_1 = explicit_unroll(
        model, inputs, targets
    )
    direct = detached_state_direct(
        direct_model, inputs, targets, exact_state_1
    )
    (
        actual,
        first_raw_0,
        replay_raw_0,
        first_raw_1,
        countdown,
        replay_state_1,
    ) = custom_replay(replay_model, inputs, targets)

    print(f"\n{label}")
    print(
        "raw0 first/exact relative error:",
        (
            (first_raw_0 - exact_raw_0).norm()
            / exact_raw_0.norm()
        ).item(),
    )
    print(
        "raw0 replay/first relative error:",
        (
            (replay_raw_0 - first_raw_0).norm()
            / first_raw_0.norm()
        ).item(),
    )
    print(
        "state1 replay/exact relative error:",
        (
            (replay_state_1 - exact_state_1).norm()
            / exact_state_1.norm()
        ).item(),
    )
    print(
        "terminal countdown relative norm:",
        (
            (countdown - first_raw_1).norm()
            / first_raw_1.norm()
        ).item(),
    )
    print(
        f"{'parameter':24s} {'exact':>12s} {'custom':>12s} "
        f"{'rel_error':>12s} {'cosine':>10s}"
    )
    for row in comparison(expected, actual):
        print(
            f"{row[0]:24s} {row[1]:12.5e} {row[2]:12.5e} "
            f"{row[3]:12.5e} {row[4]:10.6f}"
        )
    print("meta correction only")
    expected_correction = subtract_gradients(expected, direct)
    actual_correction = subtract_gradients(actual, direct)
    for row in comparison(expected_correction, actual_correction):
        print(
            f"{row[0]:24s} {row[1]:12.5e} {row[2]:12.5e} "
            f"{row[3]:12.5e} {row[4]:10.6f}"
        )


def explicit_horizon(model, inputs, targets):
    batch_size = inputs[0].shape[0]
    state = torch.zeros(
        model.layer_count,
        batch_size,
        model.fast_size,
        model.fast_size,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )
    detached_states = []
    losses = []
    for index, (x, target) in enumerate(zip(inputs, targets)):
        detached_states.append(state.detach())
        loss = loss_fn(model(x, state), target)
        losses.append(loss)
        if index != len(inputs) - 1:
            raw = torch.autograd.grad(
                loss,
                state,
                create_graph=True,
            )[0]
            state = state + normalized_update(
                raw,
                model.learning_rate(x, state),
            )
    torch.stack(losses).sum().backward()
    return named_gradients(model), detached_states


def detached_horizon_direct(model, inputs, targets, states):
    losses = [
        loss_fn(model(x, state), target)
        for x, target, state in zip(inputs, targets, states)
    ]
    torch.stack(losses).sum().backward()
    return named_gradients(model)


def custom_horizon(model, inputs, targets):
    batch_size = inputs[0].shape[0]
    state = torch.zeros(
        model.layer_count,
        batch_size,
        model.fast_size,
        model.fast_size,
        device=DEVICE,
        dtype=DTYPE,
    )
    countdown = torch.zeros_like(state, requires_grad=True)
    countdown.grad = torch.zeros_like(countdown)

    for index, (x, target) in enumerate(zip(inputs, targets)):
        loss = loss_fn(
            model(x, state, countdown),
            target,
        )
        torch.autograd.backward(loss, inputs=(countdown,))
        raw = countdown.grad.detach().clone()
        with torch.no_grad():
            countdown.add_(raw)
            if index != len(inputs) - 1:
                state.add_(
                    normalized_update(
                        raw,
                        model.learning_rate(x, state),
                    )
                )
            countdown.grad.zero_()

    state.zero_()
    model.zero_grad(set_to_none=True)
    for x, target in zip(inputs[:-1], targets[:-1]):
        propagated_lr = model.learning_rate(x, state)
        lr_leaf = propagated_lr.detach().requires_grad_(True)
        double_x = x[:, None].expand(
            -1, 2, *x.shape[1:]
        ).flatten(0, 1)
        output = model(
            double_x,
            state,
            countdown,
            countdown.detach(),
            lr_leaf,
        ).reshape(
            batch_size, 2, *target.shape[1:]
        )[:, 0]
        loss_fn(output, target).backward()
        (
            propagated_lr
            * lr_leaf.grad.detach().to(propagated_lr.dtype)
        ).sum().backward()
        raw = countdown.grad.detach().clone()
        with torch.no_grad():
            state.add_(normalized_update(raw, lr_leaf))
            countdown.sub_(raw)
            countdown.grad.zero_()

    loss_fn(
        model(inputs[-1], state, countdown),
        targets[-1],
    ).backward()
    return named_gradients(model)


def run_horizon_case(horizon, log_lr):
    torch.manual_seed(1234)
    model = StackedToyModel(log_lr=log_lr).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    custom_model = copy.deepcopy(model)
    direct_model = copy.deepcopy(model)
    inputs = [
        torch.randn(2, 5, model.hidden_size, device=DEVICE)
        for _ in range(horizon)
    ]
    targets = [
        torch.randn(2, 5, model.hidden_size, device=DEVICE)
        for _ in range(horizon)
    ]
    exact, states = explicit_horizon(model, inputs, targets)
    direct = detached_horizon_direct(
        direct_model,
        inputs,
        targets,
        states,
    )
    actual = custom_horizon(custom_model, inputs, targets)
    exact_correction = subtract_gradients(exact, direct)
    actual_correction = subtract_gradients(actual, direct)
    print(f"\nhorizon={horizon}, log_lr={log_lr:g}")
    print(
        f"{'parameter':34s} {'total err':>10s} {'total cos':>10s} "
        f"{'meta err':>10s} {'meta cos':>10s}"
    )
    total_rows = {
        row[0]: row for row in comparison(exact, actual)
    }
    correction_rows = {
        row[0]: row
        for row in comparison(
            exact_correction,
            actual_correction,
        )
    }
    for name in exact:
        total = total_rows[name]
        correction = correction_rows[name]
        print(
            f"{name:34s} {total[3]:10.4f} {total[4]:10.4f} "
            f"{correction[3]:10.4f} {correction[4]:10.4f}"
        )


def main():
    torch.manual_seed(1234)
    run_case(ToyModel(), "one layer")
    torch.manual_seed(1234)
    run_case(StackedToyModel(), "three stacked layers")
    run_horizon_case(16, -4.0)
    run_horizon_case(16, -2.5)


if __name__ == "__main__":
    main()
