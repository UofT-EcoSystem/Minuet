__all__ = [
    'ScalarOrIterable', 'ScalarOrTuple', 'SparseTensor', 'PointTensor',
    'KernelMapCache'
]

from typing import TYPE_CHECKING, TypeVar, Union, Tuple, Iterable

T = TypeVar('T')
ScalarOrTuple = Union[T, Tuple[T, ...]]
ScalarOrIterable = Union[T, Iterable[T]]

SparseTensor = PointTensor = KernelMapCache = None
if TYPE_CHECKING:
  from minuet import SparseTensor
  from minuet import PointTensor
  from minuet.nn.convolutions import KernelMapCache
