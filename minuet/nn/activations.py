__all__ = ['ReLU', 'ReLU6', 'LeakyReLU']

import torch

from minuet import SparseTensor


class ReLU(torch.nn.ReLU):
  r"""
  Applies :py:class:`torch.nn.ReLU` on the feature tensor of the given
  :py:class:`~minuet.tensors.SparseTensor` :code:`x`.
  """

  def forward(self, x: SparseTensor):
    return SparseTensor(coordinates=x.C,
                        features=super().forward(x.F),
                        stride=x.stride,
                        batch_dims=x.batch_dims)


class ReLU6(torch.nn.ReLU6):
  r"""
  Applies :py:class:`torch.nn.ReLU6` on the feature tensor of the given
  :py:class:`~minuet.tensors.SparseTensor` :code:`x`.
  """

  def forward(self, x: SparseTensor):
    return SparseTensor(coordinates=x.C,
                        features=super().forward(x.F),
                        stride=x.stride,
                        batch_dims=x.batch_dims)


class LeakyReLU(torch.nn.LeakyReLU):
  r"""
  Applies :py:class:`torch.nn.LeakyReLU` on the feature tensor of the given
  :py:class:`~minuet.tensors.SparseTensor` :code:`x`.
  """

  def forward(self, x: SparseTensor):
    return SparseTensor(coordinates=x.C,
                        features=super().forward(x.F),
                        stride=x.stride,
                        batch_dims=x.batch_dims)
