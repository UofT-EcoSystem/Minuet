__all__ = ['ReLU', 'ReLU6', 'LeakyReLU']

import torch

from minuet import SparseTensor


class ReLU(torch.nn.ReLU):

  def forward(self, x: SparseTensor):
    return SparseTensor(coordinates=x.C,
                        features=super().forward(x.F),
                        stride=x.stride,
                        batch_dims=x.batch_dims)


class ReLU6(torch.nn.ReLU6):

  def forward(self, x: SparseTensor):
    return SparseTensor(coordinates=x.C,
                        features=super().forward(x.F),
                        stride=x.stride,
                        batch_dims=x.batch_dims)


class LeakyReLU(torch.nn.LeakyReLU):

  def forward(self, x: SparseTensor):
    return SparseTensor(coordinates=x.C,
                        features=super().forward(x.F),
                        stride=x.stride,
                        batch_dims=x.batch_dims)
