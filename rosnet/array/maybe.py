from contextlib import suppress
import numpy as np
import autoray
from rosnet.core.mixin import ArrayFunctionMixin


class MaybeArray(np.lib.mixins.NDArrayOperatorsMixin, ArrayFunctionMixin):
    def __init__(self, array=None):
        self.__array = array

    @property
    def isinit(self) -> bool:
        return self.__array != None

    def __getstate__(self):
        return {"init": self.isinit, "array": self.__array}

    def __setstate__(self, d):
        self.__array = d["array"]

    def __array__(self) -> np.ndarray:
        if not self.isinit:
            raise ValueError("array not initialized")
        return self.__array

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if ufunc.nin > 2:
            return NotImplemented

        if method == "__call__":
            other = next(filter(lambda x: x == self, inputs))
            if self.isinit:
                ufunc(*inputs, **kwargs)
            else:
                if "out" in kwargs and kwargs["out"] == self:
                    self.__array = other
                else:
                    return other
        else:
            return NotImplemented
