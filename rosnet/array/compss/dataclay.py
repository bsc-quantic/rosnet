import numpy as np
from dataclay import DataClayObject, dclayMethod
from rosnet import dispatch as dispatcher


class DataClayBlock(DataClayObject):
    """A persistent block class.

    @dclayImport numpy as np
    @ClassField data numpy.ndarray
    """

    @dclayMethod(arr="numpy.ndarray")
    def __init__(self, arr):
        self.data = arr

    @dclayMethod(key="anything", return_="anything")
    def __getitem__(self, key):
        return self.data[key]

    @dclayMethod(key="anything", value="anything")
    def __setitem__(self, key, value):
        if key == ():
            self.data = value
        self.data[key] = value

    @dclayMethod(return_="numpy.ndarray")
    def __array__(self) -> np.ndarray:
        return self.data

    @dclayMethod(ufunc="numpy.ufunc", method="str", inputs="list", kwargs="dict", return_="anything")
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        "Bypasses computation to dataClay memory space. Uses numpy dispatch mechanism to call the correct implementation."
        inputs = [i.data if isinstance(i, type(self)) else i for i in inputs]
        return ufunc(*inputs, **kwargs)

    @dclayMethod(function="anything", types="list", inputs="list", kwargs="dict", return_="anything")
    def __array_function__(self, function, types, inputs, kwargs):
        "Bypasses computation to dataClay memory space. Uses numpy dispatch mechanism to call the correct implementation."
        inputs = [i.data if isinstance(i, type(self)) else i for i in inputs]
        return function(*inputs, **kwargs)


@dispatcher.to_numpy.register
def to_numpy(arr: DataClayBlock):
    return np.array(arr)
