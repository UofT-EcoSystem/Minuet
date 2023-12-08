__all__ = [
    'arg_sort_coordinates',
    'build_sorted_index',
    'query_sorted_index_with_offsets',
]

from typing import Optional

import torch

from minuet.nn.functional import _C


def arg_sort_coordinates(coordinates: torch.Tensor,
                         batch_dims: Optional[torch.Tensor] = None,
                         dtype: Optional[torch.dtype] = None,
                         enable_flattening: bool = True):
  r"""
  Sorting the coordinates in the given coordinate tensor. Multiple coordinate
  tensors can be handled together by specifying the ``batch_dims`` tensor,
  which stores the start and the end indices of each coordinate tensors. By
  default,
  `cub::DeviceRadixSort <https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html>`_
  does not support coordinate sorting. Sorting coordinates are achieved by
  :math:`N` independent launches of ``cub::DeviceRadixSort`` for
  :math:`N`-D coordinates.

  To optimize this, for coordinates with small ranges, we could compress them
  into a single ``int64`` or ``int32`` value and launch a single
  ``cub::DeviceRadixSort``. The flag ``enable_flattening`` controls whether to
  enable this optimization. If the coordinate range is too large to compress
  into the maximum possible integer type, it will fall back to the naive way of
  sorting coordinates (i.e. with :math:`N` ``cub::DeviceRadixSort`` kernel
  launches).

  Args:
    coordinates: the coordinate tensor(s) for which index is to be built
    batch_dims: the batch_dims tensor if there are multiple coordinate tensors
    enable_flattening: whether to enable coordinate flatting to speedup sorting.

  Returns:
    A tensor specifying the order of the sorted coordinates
  """

  coordinates = coordinates.contiguous()
  if coordinates.is_cuda:
    if batch_dims is not None:
      batch_dims = batch_dims.contiguous()
      index = _C.cuda_multi_arg_sort_coordinates(coordinates, batch_dims,
                                                 enable_flattening)
    else:
      index = _C.cuda_arg_sort_coordinates(coordinates, enable_flattening)
    if dtype is not None:
      index = index.to(dtype=dtype)
    return index
  raise NotImplementedError


def build_sorted_index(coordinates: torch.Tensor,
                       batch_dims: Optional[torch.Tensor] = None,
                       enable_flattening: bool = True):
  r"""
  Build sorted index for coordinate tensor(s). Multiple coordinate tensors
  can be handled together by specifying the ``batch_dims`` tensor, which stores
  the start and the end indices of each coordinate tensors.

  Args:
    coordinates: the coordinate tensor(s) for which index is to be built
    batch_dims: the batch_dims tensor if there are multiple coordinate tensors
    enable_flattening: whether to enable coordinate flatting to speedup sorting.

  Returns:
    A tensor specifying the order of the sorted coordinates

  """
  return arg_sort_coordinates(coordinates,
                              batch_dims=batch_dims,
                              dtype=torch.int64,
                              enable_flattening=enable_flattening)


def query_sorted_index_with_offsets(
    sources: torch.Tensor,
    targets: torch.Tensor,
    offsets: torch.Tensor,
    source_batch_dims: Optional[torch.Tensor] = None,
    target_batch_dims: Optional[torch.Tensor] = None):
  r"""
  Build the kernel map with the **sorted** coordinates by querying the
  sorted indices. Multiple requests can be handled together by specifying the
  ``source_batch_dims`` and ``target_batch_dims`` tensor, which stores
  the start and the end indices of each input coordinate tensor and output
  coordinate tensor respectively.

  Args:
    sources: the **sorted** input coordinates of shape
      :math:`[N_\text{in}, D]`
    targets: the **sorted** output coordinates of shape
      :math:`[N_\text{out}, D]`
    offsets: the offsets of the weights of shape :math:`[N_\text{weight}, D]`
    source_batch_dims:
      the batch_dims tensor (of the input coordinates) if there are
      multiple coordinate tensors
    target_batch_dims:
      the batch_dims tensor (of the output coordinates) if there are
      multiple coordinate tensors

  Returns:
    A tensor specifying the kernel map of shape
    :math:`[N_\text{weight}, N_\text{out}]`
  """
  if (source_batch_dims is None) != (target_batch_dims is None):
    raise ValueError("source_batch_dims and target_batch_dims must be both "
                     "provided or neither provided")
  sources = sources.contiguous()
  targets = targets.contiguous()
  offsets = offsets.contiguous()
  if source_batch_dims is not None and target_batch_dims is not None:
    source_batch_dims = source_batch_dims.contiguous()
    target_batch_dims = target_batch_dims.contiguous()
    if offsets.is_cuda:
      return _C.cuda_multi_query_sorted_index_with_offsets(
          source_batch_dims, target_batch_dims, sources, targets, offsets)
  else:
    if offsets.is_cuda:
      return _C.cuda_query_sorted_index_with_offsets(sources, targets, offsets)
  raise NotImplementedError
