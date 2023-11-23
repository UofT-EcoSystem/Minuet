__all__ = ['global_avg_pool', 'global_max_pool', 'global_sum_pool']

import torch

from minuet.utils.typing import SparseTensor


def global_avg_pool(x: SparseTensor):
  if x.batch_dims is not None:
    outputs = []
    for k in range(x.batch_size):
      features = x.F[x.batch_dims[k]:x.batch_dims[k + 1]]
      features = torch.mean(features, dim=0)
      outputs.append(features)
    outputs = torch.stack(outputs, dim=0)
  else:
    outputs = torch.mean(x.F, dim=0)
  return outputs


def global_sum_pool(x: SparseTensor):
  if x.batch_dims is not None:
    outputs = []
    for k in range(x.batch_size):
      features = x.F[x.batch_dims[k]:x.batch_dims[k + 1]]
      features = torch.sum(features, dim=0)
      outputs.append(features)
    outputs = torch.stack(outputs, dim=0)
  else:
    outputs = torch.sum(x.F, dim=0)
  return outputs


def global_max_pool(x: SparseTensor):
  if x.batch_dims is not None:
    outputs = []
    for k in range(x.batch_size):
      features = x.F[x.batch_dims[k]:x.batch_dims[k + 1]]
      features = torch.max(features, dim=0)[0]
      outputs.append(features)
    outputs = torch.stack(outputs, dim=0)
  else:
    outputs = torch.max(x.F, dim=0)[0]
  return outputs
