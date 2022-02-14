from typing import Tuple
import numpy as np


class MockArray:
    def __init__(self, shape, **kwargs):
        self.__shape = shape
        self.__dtype = kwargs.get("dtype", np.generic)

    def __array__(self):
        return np.zeros(blockshape, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented

    def __array_function__(self, function, types, args, kwargs):
        return NotImplemented

    def __getitem__(self, key):
        return 0

    @property
    def shape(self) -> Tuple[int]:
        return self.__shape

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)
