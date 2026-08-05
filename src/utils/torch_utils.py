from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import utils.constants as constants

if constants.XLA_AVAILABLE:
    import torch_xla
    from torch_xla.experimental.scan import scan as xla_scan

"""
A collection of PyTorch utility functions that might be useful.
"""


class ScannedTrainingLoop(nn.Module):
    """Run ``function`` in an XLA loop, accumulating supplied gradients."""

    def __init__(self, model: nn.Module | None, function: callable):
        super().__init__()
        object.__setattr__(self, "_scanned_model", model)
        self.function = function

    def forward(
        self,
        iterable_tensors: tuple[torch.Tensor, ...],
        *carried_tensors: torch.Tensor,
    ):
        model = object.__getattribute__(self, "_scanned_model")
        parameters = tuple(
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ) if model is not None else ()
        accumulated_gradients = []
        for parameter in parameters:
            gradient = torch.zeros_like(parameter)
            if constants.XLA_AVAILABLE:
                sharding = torch_xla._XLAC._get_xla_op_sharding(parameter)
                if sharding:
                    torch_xla._XLAC._xla_mark_sharding(gradient, sharding)
            accumulated_gradients.append(gradient)

        def step(carry, values):
            tensors, gradient_sums = carry
            value, gradients, *tensors = self.function(*values, *tensors)
            if gradients is not None:
                gradient_sums = tuple(
                    total + gradient
                    for total, gradient in zip(gradient_sums, gradients)
                )
            return (tuple(tensors), gradient_sums), value

        (carried_tensors, accumulated_gradients), values = xla_scan(
            step,
            (carried_tensors, tuple(accumulated_gradients)),
            iterable_tensors,
            is_fn_pure=True,
        )
        for parameter, gradient in zip(parameters, accumulated_gradients):
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)
        return values.sum(), *carried_tensors


def set_no_muon(model: nn.Module) -> None:
    """Mark parameters matching the model's ``no_muon_patterns``."""
    for module in model.modules():
        patterns = getattr(module, "no_muon_patterns", ())
        for name, param in module.named_parameters(recurse=True):
            if any(pattern in name for pattern in patterns):
                param.no_muon = True

    return model


def scale_gradient(
    x: torch.Tensor,
    scale: torch.Tensor | float | dict
) -> torch.Tensor:
    """
    Scales the gradient flowing through x by the given scale factor.

    If scale is a dict, it should have a "value" key containing the actual scale factor once the backward pass is executed.
    However, it can be empty during the forward pass, which allows you to set the scale factor after the forward pass is complete.
    
    Args:
        x (torch.Tensor): Input tensor.
        scale (torch.Tensor | float | dict): Scale factor for the gradient.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with the gradient scaled by scale during backpropagation.
    """
    return _ScaleGradient.apply(x, scale)

class _ScaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        if isinstance(scale, torch.Tensor):
            ctx.save_for_backward(scale)
        else:
            ctx.scale = scale
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):

        if hasattr(ctx, "scale"):
            scale = ctx.scale
        else:
            scale, = ctx.saved_tensors

        if isinstance(scale, dict):
            scale = scale["value"]

        if isinstance(scale, torch.Tensor):
            scale = scale.to(grad_output.dtype)

        return grad_output * scale, None



def attach_gradient(
    real: torch.Tensor,
    ghost: torch.Tensor,
) -> torch.Tensor:
    """
    Attaches the gradient of `ghost` to `real` during backpropagation, so that
    `ghost` will also receive the same gradient as `real` during backpropagation.

    Args:
        real (torch.Tensor): The tensor whose value is used in the forward pass.
        ghost (torch.Tensor): The tensor which also recieves the gradient during backpropagation.
    Returns:
        torch.Tensor: Tensor with the same data as real, but with the gradient of ghost attached during backpropagation.
    """
    return _AttachGradient.apply(real, ghost)

class _AttachGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, real, ghost):
        return real.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


def print_gradient(
    x: torch.Tensor,
    name: str | None = None,
) -> torch.Tensor:
    """
    Print the gradient flowing through x during backpropagation.
    
    If name is provided, it will be printed before the gradient for easier identification.
    
    Args:
        x (torch.Tensor): Input tensor.
        name (str | None): Optional name to identify the gradient.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with the gradient printed during backpropagation.
    """
    return _PrintGradient.apply(x, name)

class _PrintGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x
    
    @staticmethod
    def backward(ctx, grad_output):

        if ctx.name is not None:
            print(f"Gradient of {ctx.name}:", flush=True)
        else:
            print("Gradient:", flush=True)
        print(grad_output, flush=True)
        
        return grad_output, None


def transform_gradient(
    x: torch.Tensor,
    fn: callable,
    fn_kwargs: dict={},
) -> torch.Tensor:
    """
    Applies a transformation function to the gradient flowing through x.

    fn should have the signature: fn(x, grad_output, **fn_kwargs) -> transformed_grad_output
    
    Args:
        x (torch.Tensor): Input tensor.
        fn (callable): Function to transform the gradient.
        fn_kwargs (dict): Additional keyword arguments for the function.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with the gradient transformed by fn during backpropagation.
    """
    return _TransformGradient.apply(x, fn, fn_kwargs)

class _TransformGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, fn, fn_kwargs):
        ctx.save_for_backward(x)
        ctx.fn = fn
        ctx.fn_kwargs = fn_kwargs
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]

        grad_output = ctx.fn(
            x, grad_output, **ctx.fn_kwargs
        )

        return grad_output, None, None


def unsqueeze_to_batch(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Add leading dimensions to x (out-of-place) until it has the same number of dimensions as target.

    Args:
        x (torch.Tensor): Input tensor to be unsqueezed.
        target (torch.Tensor): Target tensor whose number of dimensions we want to match.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with leading dimensions added.
    """

    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Add leading dimensions to x (out-of-place) until it has the same number of dimensions as target.
    Then expand those leading dimensions to match the corresponding dimensions of target.

    Existing dimensions of x are not changed.

    Args:
        x (torch.Tensor): Input tensor to be expanded.
        target (torch.Tensor): Target tensor whose number of dimensions and leading dimension sizes we want to match.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with leading dimensions added and expanded.
    """
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.expand(
        *([target.shape[i] for i in range(num_unsqueeze)] + list(og_shape))
    )

    return x


def unsqueeze_to_channel(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Add trailing dimensions to x (out-of-place) until it has the same number of dimensions as target.

    Args:
        x (torch.Tensor): Input tensor to be unsqueezed.
        target (torch.Tensor): Target tensor whose number of dimensions we want to match.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with trailing dimensions added.
    """

    while x.dim() < target.dim():
        x = x[..., None]

    return x


def expand_to_channel(
    x: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Add trailing dimensions to x (out-of-place) until it has the same number of dimensions as target.
    Then expand those trailing dimensions to match the corresponding dimensions of target.

    Existing dimensions of x are not changed.

    Args:
        x (torch.Tensor): Input tensor to be expanded.
        target (torch.Tensor): Target tensor whose number of dimensions and trailing dimension sizes we want to match.
    Returns:
        torch.Tensor: Tensor with the same data as x, but with trailing dimensions added and expanded.
    """
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[..., None]
        num_unsqueeze += 1

    x = x.expand(
        *(list(og_shape) + [target.shape[i] for i in range(num_unsqueeze)])
    )

    return x


def safe_copy_state(
    src: nn.Module,
    dst: nn.Module,
    strict: bool = True,
) -> None:
    """
    Copy the state dict from src to dst, safely cloning and detaching every tensor.
    
    dst is modified in-place, and nothing is returned.

    Args:
        src (nn.Module): Source module to copy state from.
        dst (nn.Module): Destination module to copy state to.
        strict (bool): Whether to strictly enforce that the keys in src and dst match.
    """

    state = {
        k: v.clone().detach() for k, v in src.state_dict().items()
    }

    dst.load_state_dict(state, strict=strict)


def safe_finite(x: torch.Tensor, safe=False) -> torch.Tensor:
    # `torch.nan_to_num` has historically been spotty on some backends/dtypes (e.g. XLA+bfloat16).
    # `where(isfinite)` is the most portable way to guarantee non-finite values become zeros.
    if safe:
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    else:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def newton_schulz(G, steps=5, eps=1e-7, polar=False, safety=None):
    """
    Perform spectral whitening on G using Newton-Schulz iteration.

    See: https://kellerjordan.github.io/posts/muon/

    Args:
        G (torch.Tensor): Input tensor of shape [n, m].
        steps (int): Number of iterations to perform.
        eps (float): Small constant to prevent division by zero.
    Returns:
        torch.Tensor: Spectrally whitened tensor of shape [n, m].
    """
    assert G.ndim >= 2

    polar_coeffs = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        (1.875, -1.25, 0.375),
    ]
    standard_coeffs = [
        (3.4445, -4.7750,  2.0315)
    ] * 8
    
    if safety is None:
        safety = 0.5 if polar else 1.0
    coeffs = polar_coeffs if polar else standard_coeffs

    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X_f = X.float()
    X = (X_f / (X_f.norm(dim=(-2, -1), keepdim=True) + eps)).to(X.dtype)
    X = X * safety

    # Perform the NS iterations
    for t in range(steps):
        a, b, c = coeffs[t]

        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


_cuda_newton_schulz = None
def cuda_newton_schulz():
    global _cuda_newton_schulz
    if _cuda_newton_schulz is None:
        _cuda_newton_schulz = torch.compile(
            newton_schulz,
            mode="reduce-overhead",
            fullgraph=True,
        ) 
    return _cuda_newton_schulz


def select_newton_schulz():
    if constants.XLA_AVAILABLE or not torch.cuda.is_available():
        return newton_schulz
    else:
        return cuda_newton_schulz()


def shift(
    x: torch.Tensor,
    n: int,
    dim: int,
    direction: str,
    narrow: bool,
):

    zero_shape = list(x.shape)
    zero_shape[dim] = n
    z = torch.zeros(*zero_shape, device=x.device, dtype=x.dtype)

    if direction == 'right':
        if narrow:
            x = torch.narrow(x, dim, 0, x.shape[dim] - n)
        
        l = [z, x]

    elif direction == 'left':
        if narrow:
            x = torch.narrow(x, dim, n, x.shape[dim] - n)
        
        l = [x, z]

    else:
        raise ValueError(f"Invalid direction: {direction}")
    
    return torch.cat(l, dim=dim)


def gaussian_init(module: nn.Module):
    if getattr(module, "inited", False):
        return

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1/module.in_features**0.5)
        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=1)


def safe_repeat(
    x: torch.Tensor, n_repeats: int, dim: int=0
) -> torch.Tensor:
    return torch.cat(
        [x] * n_repeats,
        dim=dim
    )


def inv_softplus(x: torch.Tensor | float) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return torch.log(x.exp() - 1.0)
    return math.log(math.exp(x) - 1.0)


def unit_softplus(x: torch.Tensor | float) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return F.softplus(x) / math.log(2)
    return math.log(1 + math.exp(x)) / math.log(2)


def slerp(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: float | torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Spherically interpolate along the last dimension."""
    
    return_dtype = v0.dtype

    v0 = v0.float()
    v1 = v1.float()

    if isinstance(t, float):
        t = torch.full_like(v0[..., :1], t)
    else:
        t = t.unsqueeze(-1)
    t = unsqueeze_to_batch(t, v0).float()

    v0_norm = v0.norm(dim=-1, keepdim=True)
    v1_norm = v1.norm(dim=-1, keepdim=True)
    
    v0 = F.normalize(v0, dim=-1, eps=eps)
    v1 = F.normalize(v1, dim=-1, eps=eps)

    theta = (v0 * v1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps).acos()
    sin_theta = theta.sin()

    result = (
        v0 * ((1 - t) * theta).sin() +
        v1 * (t * theta).sin()
    ) / sin_theta.clamp_min(eps)

    # Slerp is unstable for parallel vectors, where lerp is equivalent.
    # TODO: this could be better
    lerp = (1 - t) * v0 + t * v1
    result = torch.where(sin_theta.abs() > eps, result, lerp)
    result = torch.where(
        result.norm(dim=-1, keepdim=True) > eps,
        result,
        torch.where(
            t < 0.5, v0, v1
        )
    )
    
    result = F.normalize(result, dim=-1, eps=eps)
    result = result * ((1 - t) * v0_norm + t * v1_norm)

    return result.to(return_dtype)


def einsum_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        autocast_dtype = torch.get_autocast_dtype(device_type)

        def do_cast(t):
            return (
                t.is_floating_point() and
                t.device.type == device_type and
                t.dtype is not torch.float64
            )

        if do_cast(x):
            x = x.to(autocast_dtype)
        if do_cast(weight):
            weight = weight.to(autocast_dtype)
        if bias is not None and do_cast(bias):
            bias = bias.to(autocast_dtype)

        # Match `custom_fwd(cast_inputs=...)`: after casting its floating-point
        # inputs, the decorated function executes with autocast locally disabled.
        with torch.autocast(device_type, enabled=False):
            output = torch.einsum("...i,oi->...o", x, weight)
            if bias is not None:
                output = output + bias
            return output

    output = torch.einsum("...i,oi->...o", x, weight)
    if bias is not None:
        output = output + bias
    return output


def _pure_einsum_linear_forward(
    module: nn.Linear,
    x: torch.Tensor,
) -> torch.Tensor:
    return einsum_linear(x, module.weight, module.bias)


def apply_pure_einsum_to_nn_linear(module: nn.Module) -> nn.Module:
    """Use a pure, rank-preserving einsum forward for every ``nn.Linear``.

    This preserves the modules, parameters, hooks, and state-dict keys. Unlike
    PyTorch/XLA's patched linear, the resulting forward contains only upstream
    PyTorch operations and can therefore be used inside ``PureModule``.
    """
    for child in module.modules():
        if isinstance(child, nn.Linear):
            child.forward = MethodType(_pure_einsum_linear_forward, child)
    return module


def fixed_linear(x, weight, bias=None):
    if constants.XLA_AVAILABLE:
        return einsum_linear(x, weight, bias)
    return F.linear(x, weight, bias)
