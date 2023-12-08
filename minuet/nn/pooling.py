__all__ = ['GlobalAvgPool', 'GlobalMaxPool', 'GlobalSumPool']

import torch
from torch.nn import Module

from minuet.nn import functional as F
from minuet.utils.typing import SparseTensor


class GlobalAvgPool(Module):
  r"""
  Applies :py:func:`~minuet.nn.functional.pooling.global_avg_pool` on the given
  :py:class:`~minuet.tensors.SparseTensor` :code:`x`.
  """

  def forward(self, x: SparseTensor) -> torch.Tensor:
    return F.global_avg_pool(x)


class GlobalMaxPool(Module):
  r"""
  Applies :py:func:`~minuet.nn.functional.pooling.global_max_pool` on the given
  :py:class:`~minuet.tensors.SparseTensor` :code:`x`.
  """

  def forward(self, x: SparseTensor) -> torch.Tensor:
    return F.global_max_pool(x)


class GlobalSumPool(Module):
  r"""
  Applies :py:func:`~minuet.nn.functional.pooling.global_sum_pool` on the given
  :py:class:`~minuet.tensors.SparseTensor` :code:`x`.
  """

  def forward(self, x: SparseTensor) -> torch.Tensor:
    return F.global_sum_pool(x)
