__all__ = ['BatchNorm']

import torch

from minuet import SparseTensor


class BatchNorm(torch.nn.BatchNorm1d):
  r"""
  Applies :py:class:`torch.nn.BatchNorm1d` on the feature tensor of the given
  :py:class:`~minuet.tensors.SparseTensor` :code:`x`.
  """

  def forward(self, x: SparseTensor):
    return SparseTensor(coordinates=x.C,
                        features=super().forward(x.F),
                        stride=x.stride,
                        batch_dims=x.batch_dims)
