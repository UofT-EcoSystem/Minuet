__all__ = ['SparseTensor']

from typing import Optional

import torch

from minuet.utils.helpers import as_tuple
from minuet.utils.typing import ScalarOrTuple


def _check_tensors(features: torch.Tensor,
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


class SparseTensor(object):
  r"""
  SparseTensor stores coordinates along with its features. There are several
  constraints of the coordinates for the sparse tensor:

  * The coordinates tensor should only have :math:`2` dimensions of shape
    :math:`(N, D)`. Where :math:`N` denotes the number of points and
    :math:`D` denotes the number of dimensions (typically it is just :math:`3`).
  * All coordinates should be all integers. Specifically,
    their data types should be either :code:`torch.int32` or :code:`torch.int64`.
  * All coordinates should be unique and sorted.
  * The number of features should match the number of coordinates.
  * Both features and coordinates should be on the same device.

  Note that a :py:class:`SparseTensor` can store multiple point clouds. This is
  achieved by the :py:attr:`batch_dims` tensor, where the indices of the points
  and the features of the :math:`i`-th point cloud are within the interval
  :math:`[\text{batch_dims}[i], \text{batch_dims}[i + 1])`.

  Args:
    features: the tensor that stores all features of the point clouds
    coordinates: the coordinate that stores all coordinates of the point clouds
    stride: the stride of the tensor
    batch_dims: :attr:`None` the indices of each point cloud
  """

  def __init__(self,
               features: torch.Tensor,
               coordinates: torch.Tensor,
               *,
               stride: ScalarOrTuple[int] = 1,
               batch_dims: Optional[torch.Tensor] = None):
    _check_tensors(features, coordinates)
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
    r"""
    The number of the point clouds of the current :py:class:`SparseTensor`
    """
    return 1 if self._batch_dims is None else len(self._batch_dims) - 1

  @property
  def batch_dims(self):
    r"""The batch dims of the current :py:class:`SparseTensor`"""
    return self._batch_dims

  @property
  def device(self):
    r"""Similar to :py:meth:`torch.Tensor.device`"""
    return self._coordinates.device

  @property
  def dtype(self):
    r"""
    The data type of the feature tensor of the current
    :py:class:`SparseTensor`
    """
    return self._features.dtype

  @property
  def shape(self):
    r"""
    The shape of the feature tensor of the current :py:class:`SparseTensor`
    """
    return self._features.shape

  @property
  def num_features(self):
    r"""
    The number of the feature channels of the feature tensor of the current
    :py:class:`SparseTensor`
    """
    return self._features.shape[1]

  @property
  def F(self):
    r"""
    The feature tensor of the current :py:class:`SparseTensor`
    """
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
    r"""The coordinate tensor of the current :py:class:`SparseTensor`"""
    return self._coordinates

  @property
  def stride(self):
    r"""The stride of the current :py:class:`SparseTensor`"""
    return self._stride

  @property
  def ndim(self):
    r"""
    The number of dimensions of the coordinates of the current
    :py:class:`SparseTensor`
    """
    return self._coordinates.shape[1]

  @property
  def n(self):
    r"""
    The number of coordinates of all point clouds in the current
    :py:class:`SparseTensor`
    """
    return self._coordinates.shape[0]

  def clone(self):
    r"""
    To clone the current :py:class:`SparseTensor`

    Returns:
      A cloned :py:class:`SparseTensor`
    """
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
    r"""
    Similar to applying :py:meth:`torch.Tensor.cuda` on both the coordinate and
    the feature tensors. The only catch is that ``dtype`` parameter only applies
    to the feature tensor.
    """
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
    r"""
    Similar to applying :py:meth:`torch.Tensor.cuda` on both the coordinate and
    the feature tensors
    """
    kwargs = {'device': device, 'non_blocking': non_blocking}
    self._coordinates = self._coordinates.cuda(**kwargs)
    self._features = self._features.cuda(**kwargs)
    if self._batch_dims is not None:
      self._batch_dims = self._batch_dims.cuda(**kwargs)
    return self

  def cpu(self):
    r"""
    Similar to applying :py:meth:`torch.Tensor.cpu` on both the coordinate and
    the feature tensors
    """
    self._coordinates = self._coordinates.cpu()
    self._features = self._features.cpu()
    if self._batch_dims is not None:
      self._batch_dims = self._batch_dims.cpu()
    return self

  def detach(self):
    r"""
    Similar to applying :py:meth:`torch.Tensor.detach` on the feature tensor
    """
    return SparseTensor(self._features.detach(),
                        self._coordinates,
                        stride=self._stride,
                        batch_dims=self._batch_dims)

  def detach_(self):
    r"""
    Similar to applying :py:meth:`torch.Tensor.detach_` on the feature tensor
    """
    self._features = self._features.detach_()
    return self

  def backward(self, **kwargs):
    """
    Similar to applying :py:meth:`torch.Tensor.backward` on the feature tensor
    """
    return self._features.backward(**kwargs)

  @property
  def requires_grad(self):
    """
    Similar to applying :py:attr:`torch.Tensor.requires_grad` on
    the feature tensor
    """
    return self._features.requires_grad

  @requires_grad.setter
  def requires_grad(self, value: bool):
    self._features.requires_grad = value

  def contiguous(self):
    """
    Similar to applying :py:meth:`torch.Tensor.contiguous` on both the
    coordinate and the feature tensors
    """
    self._coordinates = self._coordinates.contiguous()
    self._features = self._features.contiguous()
    if self._batch_dims is not None:
      self._batch_dims = self._batch_dims.contiguous()
    return self

  def half(self):
    """
    Similar to applying :py:meth:`torch.Tensor.half` on the feature tensor
    """
    return SparseTensor(features=self._features.half(),
                        coordinates=self._coordinates,
                        stride=self._stride,
                        batch_dims=self._batch_dims)
