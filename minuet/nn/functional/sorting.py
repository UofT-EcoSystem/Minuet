__all__ = ['arg_sort_coordinates']

from typing import Optional

import torch

from minuet.nn.functional import _C


def arg_sort_coordinates(coordinates: torch.Tensor,
                         batch_dims: Optional[torch.Tensor] = None,
                         dtype: Optional[torch.dtype] = None,
                         enable_flattening: bool = True):
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
