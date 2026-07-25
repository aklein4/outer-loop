from copy import deepcopy

import torch
import torch.nn as nn
from functorch.compile import default_partition
from torch.utils._pytree import tree_map
from torch_xla.experimental.scan import scan
from torch_xla.experimental.scan_layers import scan_layers

from torchprime.layers.sequential import HomogeneousSequential, PyTree, splat


class HomogeneousSequentialScan(HomogeneousSequential):
  supports_scan_inputs = True

  def __init__(self, *args, partition_fn=default_partition, is_layer_pure=False):
    super().__init__(*args)
    self.partition_fn = partition_fn
    self.is_layer_pure = is_layer_pure

  def forward(self, *input, **broadcasted_inputs: PyTree):
    scanned_inputs = broadcasted_inputs.pop("_scan_inputs", None)
    # `self.children()` returns an iterator over the immediate submodules, i.e.
    # the layers we want to scan over. In the `BroadcastArguments` we extend each
    # layer's return value to also output the broadcasted inputs
    # (position IDs in case of LLMs, etc). This plumbs those values across scan
    # iterations so the same values are available to all layers.
    layers = [BroadcastArguments(m) for m in self.children()]
    if len(input) == 1:
      # Handle single argument case: we don't need to call the module with a tuple.
      input = input[0]
    carry = (input, broadcasted_inputs)
    if scanned_inputs is None:
      out, _broadcasted_inputs_back = scan_layers(
        layers, carry, partition_fn=self.partition_fn,
        is_layer_pure=self.is_layer_pure
      )
    else:
      out, _broadcasted_inputs_back = _scan_layers_with_inputs(
        layers,
        carry,
        scanned_inputs,
        partition_fn=self.partition_fn,
        is_layer_pure=self.is_layer_pure,
      )
    return out


class BroadcastArguments(torch.nn.Module):
  def __init__(self, mod: nn.Module):
    super().__init__()
    self.mod = mod

  def forward(self, orig_input, broadcasted_inputs, scanned_inputs=None):
    layer_inputs = broadcasted_inputs
    if scanned_inputs is not None:
      layer_inputs = broadcasted_inputs | scanned_inputs
    out = self.mod(*splat(orig_input), **layer_inputs)
    return (out, broadcasted_inputs)


def _scan_layers_with_inputs(
  layers, input_data, scanned_inputs, partition_fn, is_layer_pure
):
  """Run layers while consuming the leading axis of `scanned_inputs`."""
  layer_count = len(layers)
  for name, value in scanned_inputs.items():
    if value.shape[0] != layer_count:
      raise ValueError(
        f"scan input {name!r} has {value.shape[0]} entries, "
        f"expected {layer_count}"
      )

  params = [
    dict(layer.named_parameters())
    for layer in layers
  ]
  buffers = [
    dict(layer.named_buffers())
    for layer in layers
  ]
  stacked_params = tree_map(
    lambda *values: torch.stack(values), *params
  )
  stacked_buffers = tree_map(
    lambda *values: torch.stack(values), *buffers
  )
  example_layer = deepcopy(layers[0])

  def one_layer(carry, layer_inputs):
    params_buffers, inputs = layer_inputs
    output = torch.func.functional_call(
      example_layer,
      params_buffers,
      (*carry, inputs),
      strict=True,
    )
    return output, None

  final_carry, _ = scan(
    one_layer,
    input_data,
    ((stacked_params, stacked_buffers), scanned_inputs),
    partition_fn=partition_fn,
    is_fn_pure=is_layer_pure,
  )
  return final_carry


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
