__all__ = [
    'cuda_time_gather', 'cuda_time_scatter', 'sparse_convolution_forward',
    'cuda_time_gemm', 'set_gemm_parallel_level'
]

from typing import Optional

import torch

from minuet.nn.functional import _C

GEMM_PARALLEL_LEVEL = 4


def set_gemm_parallel_level(level: int):
  """
  Set the parallelization level (number of CUDA streams) for executing GEMM
  operations

  Args:
    level: the parallelization level

  """
  global GEMM_PARALLEL_LEVEL
  GEMM_PARALLEL_LEVEL = level


def cuda_time_gather(weights: torch.Tensor,
                     source_masks: torch.Tensor,
                     target_masks: torch.Tensor,
                     kernel_map_sizes: torch.Tensor,
                     tile_size: int,
                     allow_shortcut_matmul: bool = False,
                     threshold: float = 0) -> float:
  """
  Benchmark the gather operation

  Args:
    weights: the weight tensors
    source_masks: the source masks from the kernel map
    target_masks: the target masks from the kernel map
    kernel_map_sizes: the sizes of each weight in the kernel map
    tile_size: the tile size for the gather operation
    allow_shortcut_matmul: whether allows shortcut of computing trivial weight
    threshold: the threshold that controls the padding of the GEMM operands

  Returns:
    the measured time of the gather operation
  """
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
  """
  Benchmark the scatter operation

  Args:
    weights: the weight tensors
    source_masks: the source masks from the kernel map
    target_masks: the target masks from the kernel map
    kernel_map_sizes: the sizes of each weight in the kernel map
    tile_size: the tile size for the scatter operation
    allow_shortcut_matmul: whether allows shortcut of computing trivial weight
    threshold: the threshold that controls the padding of the GEMM operands

  Returns:
    the measured time of the scatter operation
  """
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
  """
  Benchmark the GEMM operation

  Args:
    weights: the weight tensors
    source_masks: the source masks from the kernel map
    target_masks: the target masks from the kernel map
    kernel_map_sizes: the sizes of each weight in the kernel map
    allow_shortcut_matmul: whether allows shortcut of computing trivial weight
    parallel: the parallelization level of the GEMM operation
    threshold: the threshold that controls the padding of the GEMM operands

  Returns:
    the measured time of the GEMM operation
  """
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
                               threshold: Optional[float] = 0) -> torch.Tensor:
  """
  Executes the forward pass of a sparse convolution

  Args:
    sources: the feature tensor of the input
      :py:class:`~minuet.tensors.SparseTensor`
    weights: the weight tensor of the :py:class:`~minuet.nn.SparseConv`
    source_masks: the source masks from the kernel map
    target_masks: the target masks from the kernel map
    kernel_map_order: the order of the sorted weights by the kernel map sizes
    kernel_map_sizes: the sizes of each weight in the kernel map
    gather_tile_size: the tile size for the gather operation
    scatter_tile_size: the tile size for the scatter operation
    allow_shortcut_matmul: whether allows shortcut of computing trivial weight
    parallel: the parallelization level of the GEMM operation
    threshold: the threshold that controls the padding of the GEMM operands

  Returns:
    the feature tensor of the output :py:class:`~minuet.tensors.SparseTensor`
  """
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
