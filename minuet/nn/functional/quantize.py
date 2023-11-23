__all__ = ['quantize']

import torch

from minuet.nn import functional as F
from minuet.utils.helpers import as_tuple
from minuet.utils.typing import ScalarOrTuple


def quantize(coordinates: torch.Tensor,
             voxel_size: ScalarOrTuple[float] = 1,
             dtype: torch.dtype = torch.int32,
             return_reverse_indices: bool = False):
  voxel_size = torch.tensor(as_tuple(voxel_size),
                            dtype=coordinates.dtype,
                            device=coordinates.dtype)
  coordinates = torch.floor(coordinates / voxel_size).to(dtype)
  coordinates, reverse_indices = F.unique_coordinates(coordinates)
  if return_reverse_indices:
    return coordinates, reverse_indices
  else:
    return coordinates
