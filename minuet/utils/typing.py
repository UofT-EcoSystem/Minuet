__all__ = [
    'ScalarOrIterable', 'ScalarOrTuple', 'SparseTensor', 'KernelMapCache'
]

from typing import TYPE_CHECKING, TypeVar, Union, Tuple, Iterable

T = TypeVar('T')
ScalarOrTuple = Union[T, Tuple[T, ...]]
r"""The generic type with a type parameter ``T`` standing for the value is 
either a scalar value or a tuple consists of scalar values of type ``T``"""

ScalarOrIterable = Union[T, Iterable[T]]
r"""The generic type with a type parameter ``T`` standing for the value is 
either a scalar value or an iterable that generates scalar values of 
type ``T``"""

SparseTensor = KernelMapCache = None
if TYPE_CHECKING:
  from minuet import SparseTensor
  from minuet.nn.convolutions import KernelMapCache
