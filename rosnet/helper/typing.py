from typing import Protocol, Tuple, List, runtime_checkable
import numpy as np
from plum import parametric, type_of, ptype


@runtime_checkable
class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray:
        pass


@runtime_checkable
class Array(SupportsArray, Protocol):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        pass

    def __array_function__(self, function, types, args, kwargs):
        pass

    def shape(self) -> Tuple[int]:
        pass

    def dtype(self) -> np.dtype:
        pass
