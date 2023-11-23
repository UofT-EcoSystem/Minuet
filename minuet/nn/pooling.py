__all__ = ['GlobalAvgPool', 'GlobalMaxPool', 'GlobalSumPool']

import torch
from torch.nn import Module

from minuet.nn import functional as F
from minuet.utils.typing import SparseTensor


class GlobalAvgPool(Module):

  def forward(self, x: SparseTensor) -> torch.Tensor:
    return F.global_avg_pool(x)


class GlobalMaxPool(Module):

  def forward(self, x: SparseTensor) -> torch.Tensor:
    return F.global_max_pool(x)


class GlobalSumPool(Module):

  def forward(self, x: SparseTensor) -> torch.Tensor:
    return F.global_sum_pool(x)
