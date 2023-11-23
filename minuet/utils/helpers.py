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

from minuet.utils.file_system import ensure_directory
from minuet.utils.typing import ScalarOrTuple, KernelMapCache, SparseTensor, ScalarOrIterable


def set_kernel_map_cache(module: Module, cache: KernelMapCache):
  if hasattr(module, "set_kernel_map_cache"):
    module.set_kernel_map_cache(cache)

  for submodule in module.children():
    set_kernel_map_cache(submodule, cache=cache)

  return module


def autotune(model: Module,
             model_cache: KernelMapCache,
             data: ScalarOrIterable[SparseTensor],
             cache_path: Optional[str] = None):
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
    ensure_directory(path=os.path.dirname(cache_path))
    with open(cache_path, "w") as writer:
      json.dump(dump_tunable_config(model), writer, indent=2)

  return model


def load_tunable_config(module: Module, config: dict):
  if hasattr(module, "tunable_config"):
    module.tunable_config.update(config)

  for name, submodule in module.named_children():
    load_tunable_config(submodule, config=config[name])

  return module


def dump_tunable_config(module: Module) -> dict:
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
