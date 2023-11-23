__all__ = ['simple_hash', 'unique_coordinates']

from typing import Optional

import torch

from minuet.nn.functional import _C


def simple_hash(coordinates: torch.Tensor, reverse: bool = False):
  coordinates = coordinates.contiguous()
  if coordinates.is_cuda:
    return _C.cuda_simple_hash(coordinates, reverse)
  raise NotImplementedError


def unique_coordinates(coordinates: torch.Tensor,
                       batch_dims: Optional[torch.Tensor] = None):
  coordinates = coordinates.contiguous()
  if batch_dims is None:
    if coordinates.is_cuda:
      return _C.cuda_unique_coordinates(coordinates)
  else:
    batch_dims = batch_dims.contiguous()
    if coordinates.is_cuda:
      return _C.cuda_multi_unique_coordinates(coordinates, batch_dims)
  raise NotImplementedError
