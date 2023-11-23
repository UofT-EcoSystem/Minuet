__all__ = [
    'multi_cumsum', 'compute_kernel_map_masks', 'compute_kernel_map_sizes'
]

import torch

from minuet.nn.functional import _C


def multi_cumsum(tensor: torch.Tensor, cumsum_type: str = "inclusive"):
  tensor = tensor.contiguous()
  if tensor.is_cuda:
    return _C.cuda_multi_cumsum(tensor, cumsum_type)
  raise NotImplementedError


def compute_kernel_map_sizes(kernel_map: torch.Tensor):
  kernel_map = kernel_map.contiguous()
  if kernel_map.is_cuda:
    return _C.cuda_compute_kernel_map_sizes(kernel_map)
  raise NotImplementedError


def compute_kernel_map_masks(num_sources: int, kernel_map: torch.Tensor,
                             kernel_map_sizes: torch.Tensor):
  kernel_map = kernel_map.contiguous()
  kernel_map_sizes = kernel_map_sizes.contiguous()
  if kernel_map.is_cuda:
    return _C.cuda_compute_kernel_map_masks(num_sources, kernel_map,
                                            kernel_map_sizes)
  raise NotImplementedError
