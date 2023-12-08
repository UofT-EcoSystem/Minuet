__all__ = ['compute_kernel_map_masks', 'compute_kernel_map_sizes']

import torch

from minuet.nn.functional import _C


def compute_kernel_map_sizes(kernel_map: torch.Tensor):
  r"""
  Compute the size of each weights of the kernel map.
  **This method should not be directly used by the user.**

  Args:
    kernel_map: the tensor of shape :math:`(N_\text{weight}, D)` representing
      the kernel map

  Returns:
    a tensor of shape :math:`(N_\text{weight})` representing the size of the
      kernel map
  """
  kernel_map = kernel_map.contiguous()
  if kernel_map.is_cuda:
    return _C.cuda_compute_kernel_map_sizes(kernel_map)
  raise NotImplementedError


def compute_kernel_map_masks(num_sources: int, kernel_map: torch.Tensor,
                             kernel_map_sizes: torch.Tensor):
  r"""
  Compute the input & output masks (metadata tables) from the kernel map.
  **This method should not be directly used by the user.**

  Args:
    num_sources: the number of source coordinates
    kernel_map: the tensor of shape :math:`(N_\text{weight}, D)` representing
      the kernel map
    kernel_map_sizes: the tensor that stores the size of the kernel map
      for each weight

  Returns:
    the tensor of input masks and the tensor of output masks
  """
  kernel_map = kernel_map.contiguous()
  kernel_map_sizes = kernel_map_sizes.contiguous()
  if kernel_map.is_cuda:
    return _C.cuda_compute_kernel_map_masks(num_sources, kernel_map,
                                            kernel_map_sizes)
  raise NotImplementedError
