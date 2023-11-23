__all__ = [
    'cuda_time_gather', 'cuda_time_scatter', 'sparse_convolution_forward',
    'cuda_time_gemm', 'set_gemm_parallel_level'
]

from typing import Optional

import torch

from minuet.nn.functional import _C

GEMM_PARALLEL_LEVEL = 4


def set_gemm_parallel_level(level: int):
  global GEMM_PARALLEL_LEVEL
  GEMM_PARALLEL_LEVEL = level


def cuda_time_gather(weights: torch.Tensor,
                     source_masks: torch.Tensor,
                     target_masks: torch.Tensor,
                     kernel_map_sizes: torch.Tensor,
                     tile_size: int,
                     allow_shortcut_matmul: bool = False,
                     threshold: float = 0):
  return _C.cuda_time_gather(tile_size, allow_shortcut_matmul, threshold,
                             weights, source_masks, target_masks,
                             kernel_map_sizes)


def cuda_time_scatter(weights: torch.Tensor,
                      source_masks: torch.Tensor,
                      target_masks: torch.Tensor,
                      kernel_map_sizes: torch.Tensor,
                      tile_size: int,
                      allow_shortcut_matmul: bool = False,
                      threshold: float = 0):
  return _C.cuda_time_scatter(tile_size, allow_shortcut_matmul, threshold,
                              weights, source_masks, target_masks,
                              kernel_map_sizes)


def cuda_time_gemm(weights: torch.Tensor,
                   source_masks: torch.Tensor,
                   target_masks: torch.Tensor,
                   kernel_map_sizes: torch.Tensor,
                   allow_shortcut_matmul: bool = False,
                   parallel: Optional[int] = None,
                   threshold: float = 0):
  if parallel is None:
    parallel = GEMM_PARALLEL_LEVEL
  return _C.cuda_time_gemm(allow_shortcut_matmul, parallel, threshold, weights,
                           source_masks, target_masks, kernel_map_sizes)


def sparse_convolution_forward(sources: torch.Tensor,
                               weights: torch.Tensor,
                               source_masks: torch.Tensor,
                               target_masks: torch.Tensor,
                               kernel_map_order: Optional[torch.Tensor],
                               kernel_map_sizes: torch.Tensor,
                               gather_tile_size: int,
                               scatter_tile_size: int,
                               allow_shortcut_matmul: bool = False,
                               parallel: Optional[int] = None,
                               threshold: Optional[float] = 0):
  if sources.is_cuda:
    if parallel is None:
      parallel = GEMM_PARALLEL_LEVEL
    sources = sources.contiguous()
    weights = weights.contiguous()
    source_masks = source_masks.contiguous()
    target_masks = target_masks.contiguous()
    if kernel_map_order is not None:
      kernel_map_order = kernel_map_order.contiguous()
    kernel_map_sizes = kernel_map_sizes.contiguous()
    return _C.cuda_sparse_convolution_forward(
        gather_tile_size, scatter_tile_size, allow_shortcut_matmul, parallel,
        threshold, sources, weights, source_masks, target_masks,
        kernel_map_order, kernel_map_sizes)
  raise NotImplementedError
