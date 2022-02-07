from typing import Protocol, Tuple
import numpy as np


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
