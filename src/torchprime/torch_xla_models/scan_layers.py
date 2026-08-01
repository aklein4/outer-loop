import torch
import torch.nn as nn
from functorch.compile import default_partition
from torch_xla.experimental.scan_layers import scan_layers
from torch.utils._pytree import tree_leaves

from torchprime.layers.sequential import HomogeneousSequential, PyTree, splat


class HomogeneousSequentialScan(HomogeneousSequential):
  _next_cache_namespace = 1

  def __init__(self, *args, partition_fn=default_partition, is_layer_pure=False):
    super().__init__(*args)
    self.partition_fn = partition_fn
    self.is_layer_pure = is_layer_pure
    self._cache_namespace = HomogeneousSequentialScan._next_cache_namespace
    HomogeneousSequentialScan._next_cache_namespace += 1
    # scan_layers caches by the identity of its representative layer. Keep
    # these lightweight wrappers stable across forwards so purity caching can
    # reuse the traced forward/backward computations.
    self._scan_layer_wrappers = tuple(
      BroadcastArguments(m) for m in self.children()
    )

  def forward(self, *input, **broadcasted_inputs: PyTree):
    if len(input) == 1:
      # Handle single argument case: we don't need to call the module with a tuple.
      input = input[0]

    # PyTorch/XLA 2.8 can incorrectly reuse a pure scan computation across
    # independent stacks or across grad modes. Salt the carry shape so all
    # cache and lowering layers see these as distinct computations.
    reference = next(
      value for value in tree_leaves((input, broadcasted_inputs))
      if isinstance(value, torch.Tensor)
    )
    cache_token_size = (
      2 * self._cache_namespace + int(torch.is_grad_enabled())
    )
    cache_token = reference.new_zeros(cache_token_size)

    out, _broadcasted_inputs_back, _cache_token_back = scan_layers(
      self._scan_layer_wrappers,
      (input, broadcasted_inputs, cache_token),
      partition_fn=self.partition_fn,
      is_layer_pure=self.is_layer_pure
    )
    return out


class BroadcastArguments(torch.nn.Module):
  def __init__(self, mod: nn.Module):
    super().__init__()
    self.mod = mod

  def forward(self, orig_input, broadcasted_inputs, cache_token):
    out = self.mod(*splat(orig_input), **broadcasted_inputs)
    return (out, broadcasted_inputs, cache_token)


def compile_one_stack(
  mod: HomogeneousSequential, partition_fn=default_partition, is_layer_pure=False
) -> HomogeneousSequential:
  # Replace base class with our optimized subclass.
  if isinstance(mod, HomogeneousSequentialScan):
    raise NotImplementedError("Cannot compile HomogeneousSequential twice")
  new_mod = HomogeneousSequentialScan(*mod.children(), partition_fn=partition_fn, is_layer_pure=is_layer_pure)
  return new_mod


def compile(
  mod: nn.Module, sequential_to_scan: str, partition_fn=default_partition, is_layer_pure=False
) -> nn.Module:
  seq = mod.get_submodule(sequential_to_scan)
  if not isinstance(seq, HomogeneousSequential):
    raise ValueError(f"compile only supports HomogeneousSequential, got {type(seq)}")
  # Replace the submodule
  mod.set_submodule(
    sequential_to_scan, compile_one_stack(seq, partition_fn=partition_fn, is_layer_pure=is_layer_pure)
  )
  return mod
