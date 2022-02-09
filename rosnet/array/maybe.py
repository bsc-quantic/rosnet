from contextlib import suppress
import numpy as np
import autoray
from rosnet.helper import implements


class MaybeArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, array=None):
        self.__array = array

    @property
    def isinit(self) -> bool:
        return self.__array != None

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

    def __array_function__(self, func, types, args, kwargs):
        f = None
        with suppress(AttributeError):
            f = autoray.get_lib_fn("rosnet", str(func))

            if f is None:
                f = autoray.get_lib_fn("rosnet.MaybeArray")

        return f(*args, **kwargs) if f else NotImplemented


@implements(np.save, ext="MaybeArray")
def __maybeunitialized_save(file, arr, MaybeArray, **kwargs):
    if arr.initialized:
        np.savez(file, init=arr.initialized, array=np.array(arr))
    else:
        np.savez(file, init=arr.initialized)


@implements(np.load, ext="MaybeArray")
def __maybeunitialized_load(file, **kwargs) -> MaybeArray:
    m = np.load(file, **kwargs)
    if m["init"]:
        return MaybeArray(m["array"])
    else:
        return MaybeArray()
