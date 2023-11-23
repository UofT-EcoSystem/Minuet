__all__ = ['PointTensor', 'SparseTensor']

from typing import Optional

import torch

from minuet.utils.helpers import as_tuple
from minuet.utils.typing import ScalarOrTuple


def check_tensors(features: torch.Tensor,
                  coordinates: torch.Tensor,
                  indices: Optional[torch.Tensor] = None):
  if coordinates.ndim != 2:
    raise ValueError(f"The coordinates tensor must be a 2-D tensor but found "
                     f"{coordinates.ndim}-D")
  if coordinates.shape[0] != features.shape[0]:
    raise ValueError(f"The features tensor indicate {features.shape[0]} "
                     f"coordinates but expected {coordinates.shape[0]} "
                     f"coordinates")
  if coordinates.device != features.device:
    raise ValueError(f"The features tensor is expected to be on the same "
                     f"device as the coordinates tensor {coordinates.device} "
                     f"but found {features.device}")
  if coordinates.requires_grad:
    raise ValueError(f"The coordinates tensor should never "
                     f"requires gradients")
  if indices is not None:
    if indices.device != features.device:
      raise ValueError(f"The indices tensor is expected to be on the same "
                       f"device as the coordinates tensor {coordinates.device} "
                       f"but found {indices.device}")
    if indices.ndim != 1:
      raise ValueError(f"The indices tensor must be a 1-D tensor but found "
                       f"{indices.ndim}-D")


class PointTensor(object):

  def __init__(self, features: torch.Tensor, coordinates: torch.Tensor):
    check_tensors(features, coordinates)
    self._features = features
    self._coordinates = coordinates

  @property
  def device(self):
    return self._coordinates.device

  @property
  def dtype(self):
    return self._features.dtype

  @property
  def shape(self):
    return self._features.shape

  @property
  def F(self):
    return self._features

  @F.setter
  def F(self, features: torch.Tensor):
    if features.device != self._features.device:
      raise ValueError(f"The new features is on the device {features.device} "
                       f"which is different from expected "
                       f"{self._features.device}")
    if features.shape != self._features.shape:
      raise ValueError(f"The new features has shape {features.shape} which is "
                       f"different from expected {self._features.shape}")
    self._features = features

  @property
  def C(self):
    return self._coordinates

  @property
  def ndim(self):
    return self._coordinates.shape[1]

  @property
  def n(self):
    return self._coordinates.shape[0]

  def clone(self):
    return PointTensor(self._features.clone(), self._coordinates.clone())

  def to(self,
         device: Optional[torch.device] = None,
         dtype: Optional[torch.dtype] = None,
         non_blocking: bool = False,
         copy: bool = False):
    kwargs = {'device': device, 'non_blocking': non_blocking, 'copy': copy}
    if copy:
      coordinates = self._coordinates.to(**kwargs)
      features = self._features.to(**kwargs, dtype=dtype)
      return PointTensor(features, coordinates)

    self._coordinates = self._coordinates.to(**kwargs)
    self._features = self._features.to(**kwargs, dtype=dtype)
    return self

  def cuda(self,
           device: Optional[torch.device] = None,
           non_blocking: bool = False):
    kwargs = {'device': device, 'non_blocking': non_blocking}
    self._coordinates = self._coordinates.cuda(**kwargs)
    self._features = self._features.cuda(**kwargs)
    return self

  def cpu(self):
    self._coordinates = self._coordinates.cpu()
    self._features = self._features.cpu()
    return self

  def detach(self):
    return PointTensor(features=self._features.detach(),
                       coordinates=self._coordinates.detach())

  def detach_(self):
    self._coordinates = self._coordinates.detach_()
    self._features = self._features.detach_()
    return self


class SparseTensor(object):
  """
  SparseTensor stores coordinates along with its features. There are several
  constraints of the coordinates for the sparse tensor:

  * The coordinates tensor should only have 2 dimensions of shape (N, NDIM).
    Where N denotes the number of tensors and NDIM denotes the number of
    dimensions (typically it is just 3).
  * All coordinates should be all integers (otherwise refer to the PointTensor
    instead). Specifically, they should be either `torch.int32` or
    `torch.int64`.
  * All coordinates should be unique and sorted.
  * The number of features should match the number of coordinates.
  * Both features and coordinates should be on the same device.
  """

  def __init__(self,
               features: torch.Tensor,
               coordinates: torch.Tensor,
               *,
               stride: ScalarOrTuple[int] = 1,
               batch_dims: Optional[torch.Tensor] = None):
    check_tensors(features, coordinates)
    if coordinates.dtype not in (torch.int32, torch.int64):
      raise ValueError(f"The coordinates tensor must be int32 or int64 "
                       f"but found {coordinates.dtype}")
    self._features = features
    self._coordinates = coordinates
    self._stride = as_tuple(stride, size=self.ndim, name="strides")
    self._batch_dims = None
    if batch_dims is not None and len(batch_dims) > 2:
      self._batch_dims = batch_dims

  @property
  def batch_size(self):
    return 1 if self._batch_dims is None else len(self._batch_dims) - 1

  @property
  def batch_dims(self):
    return self._batch_dims

  @property
  def device(self):
    return self._coordinates.device

  @property
  def dtype(self):
    return self._features.dtype

  @property
  def shape(self):
    return self._features.shape

  @property
  def num_features(self):
    return self._features.shape[1]

  @property
  def F(self):
    return self._features

  @F.setter
  def F(self, features: torch.Tensor):
    if features.device != self._features.device:
      raise ValueError(f"The new features is on the device {features.device} "
                       f"which is different from expected "
                       f"{self._features.device}")
    if features.shape != self._features.shape:
      raise ValueError(f"The new features has shape {features.shape} which is "
                       f"different from expected {self._features.shape}")
    self._features = features

  @property
  def C(self):
    return self._coordinates

  @property
  def stride(self):
    return self._stride

  @property
  def ndim(self):
    return self._coordinates.shape[1]

  @property
  def n(self):
    return self._coordinates.shape[0]

  def clone(self):
    batch_dims = None
    if self._batch_dims is not None:
      batch_dims = self._batch_dims.clone()
    return SparseTensor(self._features.clone(),
                        self._coordinates.clone(),
                        batch_dims=batch_dims,
                        stride=self._stride)

  def to(self,
         device: Optional[torch.device] = None,
         dtype: Optional[torch.dtype] = None,
         non_blocking: bool = False,
         copy: bool = False):
    kwargs = {'device': device, 'non_blocking': non_blocking, 'copy': copy}
    if copy:
      coordinates = self._coordinates.to(**kwargs)
      features = self._features.to(**kwargs, dtype=dtype)
      batch_dims = None
      if self._batch_dims is not None:
        batch_dims = self._batch_dims.to(**kwargs)
      return SparseTensor(features,
                          coordinates,
                          stride=self._stride,
                          batch_dims=batch_dims)

    self._coordinates = self._coordinates.to(**kwargs)
    self._features = self._features.to(**kwargs, dtype=dtype)
    return self

  def cuda(self,
           device: Optional[torch.device] = None,
           non_blocking: bool = False):
    kwargs = {'device': device, 'non_blocking': non_blocking}
    self._coordinates = self._coordinates.cuda(**kwargs)
    self._features = self._features.cuda(**kwargs)
    if self._batch_dims is not None:
      self._batch_dims = self._batch_dims.cuda(**kwargs)
    return self

  def cpu(self):
    self._coordinates = self._coordinates.cpu()
    self._features = self._features.cpu()
    if self._batch_dims is not None:
      self._batch_dims = self._batch_dims.cpu()
    return self

  def detach(self):
    return SparseTensor(self._features.detach(),
                        self._coordinates,
                        stride=self._stride,
                        batch_dims=self._batch_dims)

  def detach_(self):
    self._features = self._features.detach_()
    return self

  def backward(self, **kwargs):
    return self._features.backward(**kwargs)

  @property
  def requires_grad(self):
    return self._features.requires_grad

  @requires_grad.setter
  def requires_grad(self, value: bool):
    self._features.requires_grad = value

  def contiguous(self):
    self._coordinates = self._coordinates.contiguous()
    self._features = self._features.contiguous()
    if self._batch_dims is not None:
      self._batch_dims = self._batch_dims.contiguous()
    return self

  def half(self):
    return SparseTensor(features=self._features.half(),
                        coordinates=self._coordinates,
                        stride=self._stride,
                        batch_dims=self._batch_dims)
