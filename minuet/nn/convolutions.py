__all__ = ['SparseConv', 'SparseConv3d', 'KernelMapCache']

import contextlib
import functools
import operator
from typing import Optional, Union, Any, Dict

import numpy as np
import torch
from torch.nn import Module

from minuet import SparseTensor
from minuet.nn import functional as F
from minuet.utils.helpers import as_tuple, generate_kernel_offsets
from minuet.utils.typing import ScalarOrTuple


class KernelMapCache(object):
  r"""
  :py:class:`KernelMapCache` builds kernel map without duplicates. It works
  jointly with :py:class:`SparseConv`.
  """

  def __init__(self,
               ndim: int,
               dtype: torch.dtype,
               device: Union[str, torch.device],
               layout: str = "minuet",
               disable_reordering: bool = False):
    self._ndim = ndim
    self._dtype = dtype
    self._device = device
    self._layout = layout
    self._kernel_maps_cache = dict()
    self._coordinates_cache = dict()
    self._sorted_coordinates_cache = dict()
    self._disable_reordering = disable_reordering

  @functools.lru_cache
  def _get_kernel_offsets(self, source_stride: ScalarOrTuple[int],
                          kernel_size: ScalarOrTuple[int],
                          kernel_dilation: ScalarOrTuple[int]):
    offsets = generate_kernel_offsets(ndim=self._ndim,
                                      kernel_size=kernel_size,
                                      source_stride=source_stride,
                                      dilation=kernel_dilation,
                                      layout=self._layout)
    return torch.tensor(offsets, dtype=self._dtype, device=self._device)

  @functools.lru_cache
  def _get_stride_tensor(self, stride: ScalarOrTuple[int]):
    return torch.tensor(stride, dtype=self._dtype, device=self._device)

  def reset(self):
    self._coordinates_cache.clear()
    self._kernel_maps_cache.clear()

  def get_target_coordinates(self,
                             source_coordinates: torch.Tensor,
                             source_batch_dims: Optional[torch.Tensor],
                             source_stride: ScalarOrTuple[int],
                             kernel_stride: ScalarOrTuple[int],
                             transposed: bool = False):
    _, ndim = source_coordinates.shape
    source_stride = as_tuple(source_stride, size=ndim, name="source_stride")
    kernel_stride = as_tuple(kernel_stride, size=ndim, name="kernel_stride")

    if source_stride not in self._coordinates_cache:
      self._coordinates_cache[source_stride] = \
        source_coordinates, source_stride, source_batch_dims

    op = operator.floordiv if transposed else operator.mul
    target_stride = tuple(
        op(a, b) for a, b in zip(source_stride, kernel_stride))

    if target_stride in self._coordinates_cache:
      return self._coordinates_cache[target_stride]

    if transposed:
      raise RuntimeError(
          "Layer must transpose=True must have its corresponding "
          "non-transposed layer before its execution")

    if target_stride not in self._coordinates_cache:
      target_stride_tensor = self._get_stride_tensor(target_stride)
      target_coordinates = source_coordinates // target_stride_tensor
      target_coordinates *= target_stride_tensor
      index = F.build_sorted_index(target_coordinates,
                                   batch_dims=source_batch_dims)
      target_coordinates = target_coordinates[index]
      target_coordinates, index = F.unique_coordinates(
          target_coordinates, batch_dims=source_batch_dims)
      target_batch_dims = source_batch_dims
      if target_batch_dims is not None:
        target_batch_dims = index[target_batch_dims.to(torch.int64)]
        target_batch_dims = target_batch_dims.to(source_batch_dims.dtype)

      self._coordinates_cache[target_stride] = \
        target_coordinates, target_stride, target_batch_dims
    return self._coordinates_cache[target_stride]

  def get_kernel_map(self,
                     source_coordinates: torch.Tensor,
                     source_stride: ScalarOrTuple[int],
                     source_batch_dims: Optional[torch.Tensor],
                     target_coordinates: torch.Tensor,
                     target_stride: ScalarOrTuple[int],
                     target_batch_dims: Optional[torch.Tensor],
                     kernel_size: ScalarOrTuple[int],
                     kernel_dilation: ScalarOrTuple[int],
                     disable_reordering: bool = False,
                     transposed: bool = False):
    num_sources, ndim = source_coordinates.shape

    source_stride = as_tuple(source_stride, size=ndim, name="source_stride")
    target_stride = as_tuple(target_stride, size=ndim, name="target_stride")

    kernel_size = as_tuple(kernel_size, size=ndim, name="kernel_size")
    kernel_dilation = as_tuple(kernel_dilation,
                               size=ndim,
                               name="kernel_dilation")

    kernel_map_key = list()
    if transposed:
      kernel_map_key.append(target_stride)
      kernel_map_key.append(source_stride)
    else:
      kernel_map_key.append(source_stride)
      kernel_map_key.append(target_stride)
    kernel_map_key.append(kernel_size)
    kernel_map_key.append(kernel_dilation)
    kernel_map_key = tuple(kernel_map_key)

    # If the kernel map is cached then just return the cached kernel map
    if kernel_map_key in self._kernel_maps_cache:
      values = list(self._kernel_maps_cache[kernel_map_key])
      if transposed:
        # Swap source_masks & target_masks
        values[0], values[1] = values[1], values[0]
      return tuple(values)

    if transposed:
      raise RuntimeError(
          "Layer must transpose=True must have its corresponding "
          "non-transposed layer before its execution")

    offsets = self._get_kernel_offsets(source_stride=source_stride,
                                       kernel_size=kernel_size,
                                       kernel_dilation=kernel_dilation)
    kernel_map = F.query_sorted_index_with_offsets(
        sources=source_coordinates,
        targets=target_coordinates,
        offsets=offsets,
        source_batch_dims=source_batch_dims,
        target_batch_dims=target_batch_dims)
    kernel_map_sizes = F.compute_kernel_map_sizes(kernel_map)
    if not self._disable_reordering and not disable_reordering:
      kernel_map_sizes, kernel_map_order = torch.sort(kernel_map_sizes)
      kernel_map = kernel_map[kernel_map_order]

      kernel_map_order = kernel_map_order.to(kernel_map_sizes.dtype)
    else:
      kernel_map_order = None

    source_masks, target_masks = F.compute_kernel_map_masks(
        num_sources=num_sources,
        kernel_map=kernel_map,
        kernel_map_sizes=kernel_map_sizes)
    self._kernel_maps_cache[kernel_map_key] = \
      source_masks, target_masks, kernel_map_order, kernel_map_sizes

    # It's impossible that transpose is True here
    return source_masks, target_masks, kernel_map_order, kernel_map_sizes


class SparseConv(Module):
  r"""
  Applies a sparse convolution over a :py:class:`~minuet.tensors.SparseTensor`.

  Each input point cloud in the :py:class:`~minuet.tensors.SparseTensor`
  consists of a set of non-zero points (i.e. the coordinate tensor of the
  :py:class:`~minuet.tensors.SparseTensor`)
  :math:`\mathcal{P} = \{\mathbf{p_j}\}` and its corresponding features
  :math:`\{\mathbf{F^\mathcal{P}_j}\}`
  (i.e. the feature tensor of the :py:class:`~minuet.tensors.SparseTensor`).
  With the stride :math:`s`, the sparse convolution will produce an output
  point cloud, where the set of non-zero points are generated as follows:

  .. math::

     \mathcal{Q} = \left\{
       \left(
         \left\lfloor \frac{x}{s} \right\rfloor \times s,
         \left\lfloor \frac{y}{s} \right\rfloor \times s,
         \left\lfloor \frac{z}{s} \right\rfloor \times s
       \right)
       ~\middle|~
       (x, y, z) \in \mathcal{P}
     \right\}

  The output feature vector :math:`\mathbf{F^\mathcal{Q}_i}` of each output
  coordinate :math:`\mathbf{q_i}` is computed as follows:

  .. math::

    \mathbf{F^\mathcal{Q}_i} =
      \sum_{\mathbf{p_j} \in \mathcal{P}}
      \sum_{\boldsymbol{\delta_k} \in \Delta}
        𝟙_{\mathbf{p_j} = \mathbf{q_i} + \boldsymbol{\delta_k}}
        \mathbf{F^\mathcal{P}_j} \mathbf{W_k}

  The :math:`\Delta` denotes the weight offsets, generated by
  :py:func:`~minuet.utils.helpers.generate_kernel_offsets`.

  Shape:
    If there are :math:`N` input points and :math:`M` output points, then:

    * Input coordinate tensor: :math:`[N, 3]`
    * Input feature tensor: :math:`[N, C_\text{in}]`
    * Output coordinate tensor: :math:`[M, 3]`
    * Output feature tensor: :math:`[M, C_\text{out}]`
    * Kernel weight offsets: :math:`[K^3, 3]`
    * Kernel weights: :math:`[K^3, C_\text{in}, C_\text{out}]`
    * Bias (if enabled): :math:`[C_\text{out}]`

  Args:
    ndim: the number of coordinate dimensions
    in_channels: the number of input feature channels :math:`C_\text{in}`
    out_channels: the number of input feature channels :math:`C_\text{out}`
    kernel_size: the kernel size of the sparse convolution :math:`K`
    stride: the stride of the sparse convolution :math:`s`
    dilation: the dilation of the sparse convolution
    bias: whether enables bias weight in the sparse convolution
    transposed: whether enables transposed sparse convolution
    dtype: the data type of the weights
    device: the device to store the weights
  """

  # H < 0:    Do not fuse any
  # H = None: Fuse all
  __AVAILABLE_THRESHOLD__ = [-1, *list(np.linspace(0, 2, num=50)), None]
  __NUM_TUNING_ROUNDS__ = 10

  def __init__(self,
               ndim: int,
               in_channels: int,
               out_channels: int,
               kernel_size: ScalarOrTuple[int],
               stride: ScalarOrTuple[int] = 1,
               dilation: ScalarOrTuple[int] = 1,
               bias: bool = False,
               transposed: bool = False,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None):
    super().__init__()
    self.ndim = ndim
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = as_tuple(kernel_size, size=ndim, name="kernel_size")
    self.stride = as_tuple(stride, size=ndim, name="stride")
    self.dilation = as_tuple(dilation, size=ndim, name="dilation")
    self.transposed = transposed

    kwargs = {'device': device, 'dtype': dtype}

    kernel_volume = np.prod(self.kernel_size).item()
    self.kernel = torch.nn.Parameter(
        torch.empty(kernel_volume, in_channels, out_channels, **kwargs))

    if bias:
      self.bias = torch.nn.Parameter(torch.empty(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

    self._kernel_map_cache: Optional[KernelMapCache] = None
    self._tunable_config = {
        'threshold': -1,
        'gather_tile_size': min(in_channels, 1),
        'scatter_tile_size': min(out_channels, 1)
    }

    self._tuning_data = None

  @property
  def _allow_shortcut_matmul(self):
    return (all(x == 1 for x in self.dilation)
            and all(x == 1 for x in self.stride)
            and not self._kernel_map_cache._disable_reordering)

  @property
  def tunable_config(self) -> Dict[str, Any]:
    r"""
    Returns:
      all tunable configurations for the :py:class:`SparseConv`
    """
    return self._tunable_config

  def extra_repr(self) -> str:
    s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
    if any(x != 1 for x in self.stride):
      s += ', stride={stride}'
    if any(x != 1 for x in self.dilation):
      s += ', dilation={dilation}'
    if self.bias is None:
      s += ', bias=False'
    if self.transposed:
      s += ', transposed=True'
    return s.format(**self.__dict__)

  def reset_parameters(self):
    r"""
    Randomly initialize all weights for the :py:class:`SparseConv`
    """
    std = (self.out_channels if self.transposed else self.in_channels)
    std = 1. / np.sqrt(std * len(self.kernel))
    self.kernel.data.uniform_(-std, std)
    if self.bias is not None:
      self.bias.data.uniform_(-std, std)

  def set_kernel_map_cache(self, cache: KernelMapCache):
    r"""
    Set the kernel map cache for the current :py:class:`SparseConv`

    Args:
      cache: the cache to be attached to the current :py:class:`SparseConv`

    """
    self._kernel_map_cache = cache

  @property
  def is_trivial(self):
    return all(x == 1 for x in self.stride) and \
      all(x == 1 for x in self.dilation) and \
      all(x == 1 for x in self.kernel_size)

  @contextlib.contextmanager
  def autotune(self, free_buffers: bool = True):
    r"""
    A context for autotuning the current :py:class:`SparseConv`.

    Examples:
      .. code-block:: python

         conv = SparseConv(...)
         with conv.autotune():
           for inputs in data_loader:
             _ = conv(inputs)

    Args:
      free_buffers: whether to release all buffers after autotuning for
        saving GPU memory
    """
    try:
      self._tuning_data = list()
      if free_buffers:
        F.cuda_free_buffers()

      yield

      if free_buffers:
        F.cuda_free_buffers()

      if self.is_trivial:
        return

      optimal_time = None
      for threshold in self.__AVAILABLE_THRESHOLD__:
        try:
          timings = []
          for _ in range(self.__NUM_TUNING_ROUNDS__):
            for source_masks, target_masks, kernel_map_sizes in self._tuning_data:
              timings.append(
                  F.cuda_time_gemm(
                      threshold=threshold,
                      weights=self.kernel,
                      source_masks=source_masks,
                      target_masks=target_masks,
                      allow_shortcut_matmul=self._allow_shortcut_matmul,
                      kernel_map_sizes=kernel_map_sizes))
          timings = np.average(timings)
          if optimal_time is None or optimal_time > timings:
            self._tunable_config['threshold'] = threshold
            optimal_time = timings
        except RuntimeError:
          F.cuda_reset_error()
          print(f"Threshold {threshold} overflows memory exiting tuning")
          print(f"optimal={self._tunable_config['threshold']}")
          break

      if free_buffers:
        F.cuda_free_buffers()

      optimal_time = None
      for tile_size in range(1, self.in_channels + 1):
        if self.in_channels % tile_size != 0:
          continue

        timings = []
        for _ in range(self.__NUM_TUNING_ROUNDS__):
          for source_masks, target_masks, kernel_map_sizes in self._tuning_data:
            timings.append(
                F.cuda_time_gather(
                    threshold=self._tunable_config['threshold'],
                    tile_size=tile_size,
                    weights=self.kernel,
                    source_masks=source_masks,
                    target_masks=target_masks,
                    allow_shortcut_matmul=self._allow_shortcut_matmul,
                    kernel_map_sizes=kernel_map_sizes))
        timings = np.average(timings)
        if optimal_time is None or optimal_time > timings:
          self._tunable_config['gather_tile_size'] = tile_size
          optimal_time = timings

      if free_buffers:
        F.cuda_free_buffers()

      optimal_time = None
      for tile_size in range(1, self.out_channels + 1):
        if self.out_channels % tile_size != 0:
          continue

        timings = []
        for _ in range(self.__NUM_TUNING_ROUNDS__):
          for source_masks, target_masks, kernel_map_sizes in self._tuning_data:
            timings.append(
                F.cuda_time_scatter(
                    threshold=self._tunable_config['threshold'],
                    tile_size=tile_size,
                    weights=self.kernel,
                    source_masks=source_masks,
                    target_masks=target_masks,
                    allow_shortcut_matmul=self._allow_shortcut_matmul,
                    kernel_map_sizes=kernel_map_sizes))
        timings = np.average(timings)
        if optimal_time is None or optimal_time > timings:
          self._tunable_config['scatter_tile_size'] = tile_size
          optimal_time = timings

      if free_buffers:
        F.cuda_free_buffers()
    finally:
      self._tuning_data = None

  def forward(self, inputs: SparseTensor):
    source_features = inputs.F
    if self.is_trivial:
      # Bypass trivial convolutions
      kernel = self.kernel.view(self.in_channels, self.out_channels)
      target_features = torch.mm(source_features, kernel)
      target_coordinates = inputs.C
      target_batch_dims = inputs.batch_dims
      target_stride = inputs.stride
    else:
      if self._kernel_map_cache is None:
        raise ValueError("Kernel map cache must be specified")

      target_coordinates, target_stride, target_batch_dims = \
        self._kernel_map_cache.get_target_coordinates(
          transposed=self.transposed,
          source_coordinates=inputs.C,
          source_batch_dims=inputs.batch_dims,
          source_stride=inputs.stride,
          kernel_stride=self.stride)

      source_masks, target_masks, kernel_map_order, kernel_map_sizes = \
        self._kernel_map_cache.get_kernel_map(
          transposed=self.transposed,
          source_coordinates=inputs.C,
          source_stride=inputs.stride,
          source_batch_dims=inputs.batch_dims,
          target_coordinates=target_coordinates,
          target_stride=target_stride,
          target_batch_dims=target_batch_dims,
          kernel_size=self.kernel_size,
          kernel_dilation=self.dilation)

      if self._tuning_data is not None:
        self._tuning_data.append((source_masks, target_masks, kernel_map_sizes))

      target_features = F.sparse_convolution_forward(
          gather_tile_size=self._tunable_config["gather_tile_size"],
          scatter_tile_size=self._tunable_config["scatter_tile_size"],
          threshold=self._tunable_config["threshold"],
          sources=source_features,
          weights=self.kernel,
          allow_shortcut_matmul=self._allow_shortcut_matmul,
          source_masks=source_masks,
          target_masks=target_masks,
          kernel_map_order=kernel_map_order,
          kernel_map_sizes=kernel_map_sizes)

    if self.bias is not None:
      target_features += self.bias
    return SparseTensor(features=target_features,
                        coordinates=target_coordinates,
                        stride=target_stride,
                        batch_dims=target_batch_dims)


class SparseConv3d(SparseConv):
  r"""
  Applies a 3-D sparse convolution over an input point cloud and produces
  an output point cloud. Please refer to :py:class:`SparseConv` for the meanings
  of each parameter.
  """

  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: ScalarOrTuple[int],
               stride: ScalarOrTuple[int] = 1,
               dilation: ScalarOrTuple[int] = 1,
               bias: bool = False,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None):
    super().__init__(ndim=3,
                     in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     dilation=dilation,
                     bias=bias,
                     transposed=False,
                     dtype=dtype,
                     device=device)
