__all__ = ['global_avg_pool', 'global_max_pool', 'global_sum_pool']

import torch

from minuet.tensors import SparseTensor


def global_avg_pool(x: SparseTensor):
  """
  Applies average pooling on the given :py:class:`~minuet.tensors.SparseTensor`.
  For each point cloud in the given :py:class:`~minuet.tensors.SparseTensor`,
  the feature tensor will be reduced with :py:func:`torch.mean` function.
  If there are multiple point clouds in the
  :py:class:`~minuet.tensors.SparseTensor`, the outputs will be stacked together
  with the batch order.

  Args:
    x: the given :py:class:`~minuet.tensors.SparseTensor` for average pooling

  Returns:
    the result tensor after average pooling

  """
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
  """
  Applies sum pooling on the given :py:class:`~minuet.tensors.SparseTensor`.
  For each point cloud in the given :py:class:`~minuet.tensors.SparseTensor`,
  the feature tensor will be reduced with :py:func:`torch.sum` function.
  If there are multiple point clouds in the
  :py:class:`~minuet.tensors.SparseTensor`, the outputs will be stacked together
  with the batch order.

  Args:
    x: the given :py:class:`~minuet.tensors.SparseTensor` for sum pooling

  Returns:
    the result tensor after sum pooling

  """
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
  """
  Applies max pooling on the given :py:class:`~minuet.tensors.SparseTensor`.
  For each point cloud in the given :py:class:`~minuet.tensors.SparseTensor`,
  the feature tensor will be reduced with :py:func:`torch.max` function.
  If there are multiple point clouds in the
  :py:class:`~minuet.tensors.SparseTensor`, the outputs will be stacked together
  with the batch order.

  Args:
    x: the given :py:class:`~minuet.tensors.SparseTensor` for average pooling

  Returns:
    the result tensor after average pooling

  """
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
