from typing import Protocol, Tuple, List
import numpy as np
from plum import parametric, type_of, ptype


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


@type_of.dispatch
def type_of(x: np.ndarray):
    level = 0
    while isinstance(x.flat[0], np.ndarray):
        level += 1
        x = x.flat[0]

    # TODO waiting for https://github.com/wesselb/plum/issues/37
    return ptype(NestedArray[level])


@parametric(runtime_type_of=True)
class NestedList(list):
    """A type for recursive lists where the type parameter specifies the nesting level."""


@type_of.dispatch
def type_of(x: List):
    level = 0
    while isinstance(x[0], List):
        level += 1
        x = x[0]

    typ = type(x[0])

    # TODO waiting for https://github.com/wesselb/plum/issues/37
    return ptype(NestedList[typ, level])
