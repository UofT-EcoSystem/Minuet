__all__ = ['cat']

from typing import Iterable

import torch
from minuet import SparseTensor


def cat(tensors: Iterable[SparseTensor]) -> SparseTensor:
  tensors = list(tensors)
  features = torch.cat([tensor.F for tensor in tensors], dim=1)
  return SparseTensor(coordinates=tensors[0].C,
                      features=features,
                      stride=tensors[0].stride,
                      batch_dims=tensors[0].batch_dims)
