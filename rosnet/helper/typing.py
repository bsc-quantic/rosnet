from typing import Protocol, Tuple
import numpy as np
from plum import parametric, type_of


class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray:
        pass


class Array(SupportsArray):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        pass

    def __array_function__(self, function, types, args, kwargs):
        pass

    @property
    def shape(self) -> Tuple[int]:
        pass

    @property
    def size(self) -> int:
        pass

    @property
    def itemsize(self) -> int:
        pass

    @property
    def nbytes(self) -> int:
        pass

    @property
    def ndim(self) -> int:
        pass

    @property
    def dtype(self) -> np.dtype:
        pass


@parametric(runtime_type_of=True)
class NestedArray(np.ndarray):
    """A type for recursive numpy arrays (array of arrays) where the type parameter specifies the nesting level."""

    pass


@type_of.dispatch
def type_of(x: np.ndarray):
    level = 0
    while isinstance(x.flat[0], np.ndarray):
        level += 1
        x = x.flat[0]

    return NestedArray[level]
