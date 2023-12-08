__all__ = ['simple_hash', 'unique_coordinates']

from typing import Optional

import torch

from minuet.nn.functional import _C


def simple_hash(coordinates: torch.Tensor,
                reverse: bool = False) -> torch.Tensor:
  r"""
  Compute the hash values of the coordinates

  Args:
    coordinates: the coordinates to be hashed
    reverse: hash each coordinates with the reverse order

  Returns:
    the hashed values for each coordinates
  """
  coordinates = coordinates.contiguous()
  if coordinates.is_cuda:
    return _C.cuda_simple_hash(coordinates, reverse)
  raise NotImplementedError


def unique_coordinates(coordinates: torch.Tensor,
                       batch_dims: Optional[torch.Tensor] = None):
  r"""
  Remove duplicated in the **sorted** coordinates. Multiple coordinate tensors
  can be handled together by specifying the ``batch_dims`` tensor, which stores
  the start and the end indices of each coordinate tensors.

  Args:
    coordinates: the tensors to be handled
    batch_dims: the batch_dims tensor if there are multiple coordinate tensors

  Returns:
    The coordinates tensor with duplicates removed
  """
  coordinates = coordinates.contiguous()
  if batch_dims is None:
    if coordinates.is_cuda:
      return _C.cuda_unique_coordinates(coordinates)
  else:
    batch_dims = batch_dims.contiguous()
    if coordinates.is_cuda:
      return _C.cuda_multi_unique_coordinates(coordinates, batch_dims)
  raise NotImplementedError
