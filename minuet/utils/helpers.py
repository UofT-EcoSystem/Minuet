__all__ = [
    'as_tuple', 'generate_kernel_offsets', 'set_kernel_map_cache',
    'dump_tunable_config', 'load_tunable_config', 'autotune'
]

import contextlib
import json
import os.path
from typing import Iterable, Optional

import numpy as np
from torch.nn import Module

import minuet
from minuet.utils.typing import ScalarOrTuple, ScalarOrIterable


def _ensure_directory(path, create_if_not_exist=True):
  if not os.path.exists(path):
    if create_if_not_exist:
      os.makedirs(path)
    else:
      raise RuntimeError(f"Directory {path} does not exist")
  elif not os.path.isdir(path):
    raise RuntimeError(f"Path {path} is not a directory")
  return path


def set_kernel_map_cache(module: Module, cache: 'minuet.nn.KernelMapCache'):
  r"""
  Set kernel map cache for a given :py:class:`torch.nn.Module` recursively.

  Args:
    module: a given :py:class:`torch.nn.Module` instance
    cache: a :py:class:`~minuet.nn.convolutions.KernelMapCache` instance

  Returns:
    the original :py:class:`torch.nn.Module` instance
  """
  if hasattr(module, "set_kernel_map_cache"):
    module.set_kernel_map_cache(cache)

  for submodule in module.children():
    set_kernel_map_cache(submodule, cache=cache)

  return module


def autotune(model: Module,
             model_cache: 'minuet.nn.KernelMapCache',
             data: ScalarOrIterable['minuet.SparseTensor'],
             cache_path: Optional[str] = None):
  r"""
  Autotune a given :py:class:`torch.nn.Module` recursively.

  Args:
    model: a given :py:class:`torch.nn.Module` instance
    model_cache: a :py:class:`~minuet.nn.convolutions.KernelMapCache` instance
    data: an iterable instance that generates
      :py:class:'~minuet.tensors.SparseTensor'
    cache_path: the path to which the autotuned configurations will be stored

  Returns:
    the original :py:class:`torch.nn.Module` instance
  """
  from minuet import SparseTensor
  if isinstance(data, SparseTensor):
    data = [data]

  if cache_path is not None and os.path.exists(cache_path):
    with open(cache_path) as reader:
      return load_tunable_config(model, config=json.load(reader))

  with contextlib.ExitStack() as stack:

    def _tune_module(module: Module):
      if hasattr(module, "autotune"):
        stack.enter_context(module.autotune())

      for submodule in module.children():
        _tune_module(submodule)

    _tune_module(model)

    for tensor in data:
      model_cache.reset()
      _ = model(tensor)

  if cache_path is not None:
    _ensure_directory(path=os.path.dirname(cache_path))
    with open(cache_path, "w") as writer:
      json.dump(dump_tunable_config(model), writer, indent=2)

  return model


def load_tunable_config(module: Module, config: dict):
  r"""
  Load tunable configurations for a :py:class:`torch.nn.Module` recursively
  from a dict

  Args:
    module: a given :py:class:`torch.nn.Module` instance
    config: the configuration dict

  Returns:
    the original :py:class:`torch.nn.Module` instance
  """
  if hasattr(module, "tunable_config"):
    module.tunable_config.update(config)

  for name, submodule in module.named_children():
    load_tunable_config(submodule, config=config[name])

  return module


def dump_tunable_config(module: Module) -> dict:
  r"""
  Dump tunable configurations of a :py:class:`torch.nn.Module` recursively
  to a dict

  Args:
    module: a given :py:class:`torch.nn.Module` instance

  Returns:
    the configuration dict
  """
  if hasattr(module, "tunable_config"):
    return module.tunable_config
  config = dict()
  for name, submodule in module.named_children():
    config[name] = dump_tunable_config(submodule)

  return config


def as_tuple(value: ScalarOrTuple,
             *,
             size: int = 3,
             name: Optional[str] = None):
  r"""
  Make a given value as a tuple of size :code:`size`

  Args:
    value: The value for making a tuple
    size: The size of the tuple
    name: The name of the tuple. Useful for showing exceptions.

  Returns:
    A tuple of size :code:`size`.
  """
  name = name or "Value"
  if not isinstance(value, Iterable):
    value = tuple(value for _ in range(size))
    return value

  value = tuple(value)
  if len(value) != size:
    raise ValueError(f"{name} must be a scalar or a iterable of size {size} "
                     f"but found {len(value)} elements")
  return value


def generate_kernel_offsets(ndim: int,
                            kernel_size: ScalarOrTuple[int],
                            source_stride: ScalarOrTuple[int],
                            dilation: ScalarOrTuple[int],
                            layout: str = "minuet"):
  r"""
  Generate the kernel weight offsets for :py:class:`~minuet.nn.SparseConv`

  Args:
    ndim: the dimension of each weight offset
    kernel_size: the kernel size of :py:class:`~minuet.nn.SparseConv`
    source_stride: the tensor stride of the input
    dilation: the dilation of :py:class:`~minuet.nn.SparseConv`
    layout: the layout of the sparse convolution, only ``minuet``,
      ``minkowski``, and ``torchsparse`` are supported

  Returns:
    a numpy array consists of the generated sparse convolution
  """
  kernel_size = as_tuple(kernel_size, size=ndim, name="kernel_size")
  source_stride = as_tuple(source_stride, size=ndim, name="source_stride")
  dilation = as_tuple(dilation, size=ndim, name="dilation")
  offsets = []
  for i in range(ndim):
    if kernel_size[i] % 2 == 1:
      k_max = kernel_size[i] // 2
      k_min = -k_max
    else:
      k_max = kernel_size[i] // 2
      k_min = -k_max + 1
    offsets.append(np.arange(k_min, k_max + 1))
  if layout == "minuet":
    offsets = np.meshgrid(*offsets, copy=False, indexing="ij")
    offsets = np.stack(offsets, axis=ndim)
  elif layout == "torchsparse":
    offsets = np.meshgrid(*offsets, copy=False, indexing="ij")
    offsets = np.stack(offsets, axis=ndim)
    if np.prod(kernel_size) % 2 == 1:
      offsets = np.transpose(offsets,
                             axes=list(range(ndim - 1, -1, -1)) + [ndim])
  elif layout == "minkowski":
    offsets = np.meshgrid(*offsets, copy=False, indexing="ij")
    offsets = np.stack(offsets, axis=ndim)
    offsets = np.transpose(offsets, axes=list(range(ndim - 1, -1, -1)) + [ndim])
  else:
    raise NotImplementedError
  offsets = np.reshape(offsets, newshape=(-1, ndim))
  return offsets * np.asarray(source_stride) * np.asarray(dilation)
