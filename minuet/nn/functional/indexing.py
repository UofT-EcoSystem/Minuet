__all__ = [
    'build_sorted_index',
    'query_sorted_index_with_offsets',
]

from typing import Optional

import torch

from minuet.nn.functional import _C
from minuet.utils.nvtx import nvtx

COMBINED_TENSOR_CACHE = dict()
SORTED_INDEX_CACHE = dict()


@nvtx("lookup_combined_coordinates")
def lookup_combined_coordinates(sources: torch.Tensor,
                                batch_dims: Optional[torch.Tensor],
                                style: str = "last_batch_dim"):
  key = (sources, batch_dims)
  if key not in COMBINED_TENSOR_CACHE:
    targets = torch.zeros([sources.shape[0], 1],
                          device=sources.device,
                          dtype=sources.dtype)
    if batch_dims is not None:
      for i in range(len(batch_dims) - 1):
        targets[batch_dims[i]:batch_dims[i + 1], -1] = i
    if style == "last_batch_dim":
      targets = torch.concat([sources, targets], dim=1)
    else:
      targets = torch.concat([targets, sources], dim=1)
    COMBINED_TENSOR_CACHE[key] = targets
  return COMBINED_TENSOR_CACHE[key]


@nvtx("lookup_sorted_index")
def lookup_sorted_index(coordinates: torch.Tensor,
                        batch_dims: Optional[torch.Tensor]):
  key = (coordinates, batch_dims)
  if key not in SORTED_INDEX_CACHE:
    SORTED_INDEX_CACHE[key] = build_sorted_index(coordinates,
                                                 batch_dims=batch_dims)
  return SORTED_INDEX_CACHE[key]


def build_sorted_index(coordinates: torch.Tensor,
                       batch_dims: Optional[torch.Tensor] = None,
                       enable_flattening: bool = True):
  from minuet.nn.functional import arg_sort_coordinates
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
