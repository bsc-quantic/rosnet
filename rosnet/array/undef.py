import numpy as np
from rosnet.helper import implements


class MaybeUnitializedArray(np.lib.mixins.NDArrayOperatorsMixin):
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
        try:
            # use autoray for dispatching
            # TODO use wrap for specialization?
            f = autoray.get_lib_fn("rosnet", str(func))

            # multiple dispatch specialization takes place here with plum
            return f(*args, **kwargs)

        except AttributeError:
            return NotImplemented


@implements(np.save, ArrayWrapper)
def __maybeunitialized_save(file, arr: ArrayWrapper(), **kwargs):
    if arr.initialized:
        np.savez(file, init=arr.initialized, array=np.array(arr))
    else:
        np.savez(file, init=arr.initialized)


@implements(np.load, ArrayWrapper)
def __maybeunitialized_load(file, **kwargs):
    m = np.load(file, **kwargs)
    if m["init"]:
        return ArrayWrapper(m["array"])
    else:
        return ArrayWrapper()
