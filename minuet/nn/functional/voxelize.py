__all__ = ['voxelize']

import torch

from minuet.nn import functional as F
from minuet.utils.helpers import as_tuple
from minuet.utils.typing import ScalarOrTuple


def voxelize(coordinates: torch.Tensor,
             voxel_size: ScalarOrTuple[float] = 1,
             dtype: torch.dtype = torch.int32,
             return_reverse_indices: bool = False):
  r"""
  Voxelize the coordinates with the given ``voxel_size``

  Args:
    coordinates: the coordinate tensor to be voxelized
    voxel_size: the size of each voxel
    dtype: the data type of for the output coordinate tensor
    return_reverse_indices: whether to return the reverse indices tensor

  Returns:
    the voxelized coordinates and (optionally) the reverse indices tensor if
    ``return_reverse_indices`` is ``True``
  """
  voxel_size = torch.tensor(as_tuple(voxel_size),
                            dtype=coordinates.dtype,
                            device=coordinates.dtype)
  coordinates = torch.floor(coordinates / voxel_size).to(dtype)
  coordinates, reverse_indices = F.unique_coordinates(coordinates)
  if return_reverse_indices:
    return coordinates, reverse_indices
  else:
    return coordinates
