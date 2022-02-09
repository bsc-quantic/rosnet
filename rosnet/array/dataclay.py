from typing import Type, Tuple
from contextlib import suppress
from math import prod
import numpy as np
from plum import dispatch
from dataclay import DataClayObject, dclayMethod
import autoray
from rosnet.helper.math import result_shape
from rosnet.helper.typing import Array, SupportsArray


class DataClayArray(np.lib.mixins.NDArrayOperatorsMixin):
    @dispatch
    def __init__(self, arr: DataClayBlock, shape: Tuple[int], dtype: np.dtype):
        self.data = arr
        self.__shape = shape
        self.__dtype = dtype

    @dispatch
    def __init__(self, arr: SupportsArray):
        "Constructor for generic arrays."
        arr = np.array(arr)
        data = DataClayBlock(arr)
        data.make_persistent()

        self.__init__(data, arr.shape, arr.dtype)

    @classmethod
    def __infer_type_parameter__(cls, *args, **kwargs) -> Type:
        return np.ndarray

    @property
    def shape(self) -> Tuple[int]:
        return self.__shape

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    def __deepcopy__(self, memo):
        raise NotImplementedError()

    def __array__(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def __array_priority__(self) -> int:
        # NOTE higher priority than COMPSsArray if conflict, but no situation like this should happen
        return 2

    def __array_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        raise NotImplementedError()

    def __array_function__(self, func, types, args, kwargs):
        f = None
        with supress(AttributeError):
            f = autoray.get_lib_fn("rosnet", func.__name__)

            if f is None:
                f = autoray.get_lib_fn("rosnet.DataClayArray", func.__name__)

            if f is None:
                f = autoray.get_lib_fn("rosnet.DataClayArray.random", func.__name__)

        return f(*args, **kwargs) if f else NotImplemented


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
    def __setitem__(self, key):
        self.data[key] = value

    @dclayMethod(returns_="numpy.ndarray")
    def __array__(self) -> np.ndarray:
        return self.data

    # TODO variable return type. maybe nothing.
    # TODO how to annotate arguments?
    @dclayMethod(...)
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        raise NotImplementedError()
        return ufunc(*inputs, **kwargs)

    # TODO variable return type. maybe nothing.
    # TODO how to annotate arguments?
    @dclayMethod(...)
    def __array_function__(self, function, types, inputs, kwargs):
        return function(*inputs, **kwargs)


@dispatch
def to_numpy(arr: BlockArray[DataClayArray]):
    return NotImplementedError()


@dispatch
def reshape(a: DataClayArray, shape, order="F", inplace=True):
    raise NotImplementedError()


@dispatch
def transpose(a: DataClayArray, axes=None, inplace=True):
    raise NotImplementedError()


@dispatch
def stack(arrays: Sequence[DataClayArray], axis=0, out=None) -> DataClayArray:
    pass


@dispatch
def split(array: DataClayArray, indices_or_sections, axis=0) -> List[DataClayArray]:
    pass


@dispatch
def tensordot(a: DataClayArray, b: DataClayArray, axes) -> DataClayArray:
    dtype = np.result_type(a.dtype, b.dtype)
    shape = result_shape(a.shape, b.shape, axes)

    raise NotImplementedError()


@dispatch
def tensordot(a: BlockArray[DataClayArray], b: BlockArray[DataClayArray], axes) -> BlockArray[DataClayArray]:
    raise NotImplementedError()
