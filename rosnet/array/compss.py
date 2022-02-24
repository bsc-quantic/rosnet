from typing import Tuple, Type, Optional, Sequence
import functools
from contextlib import suppress
from copy import deepcopy
from math import prod
import numpy as np
from multimethod import multimethod
import autoray
from pycompss.runtime.management.classes import Future as COMPSsFuture
from pycompss.api.api import compss_delete_object, compss_wait_on
from rosnet.helper.macros import todo, implements
from rosnet.helper.math import result_shape
from rosnet.helper.typing import Array, SupportsArray
from rosnet import task, tuning, numpy_interface as iface
from rosnet.array.maybe import MaybeArray


# TODO special variation for in-place functions? keep np.reshape/transpose/... as non-modifying -> create new COMPSsArray/BlockArray
# TODO support more properties of ndarray
class COMPSsArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Reference to a `numpy.ndarray` managed by COMPSs.

    Unlike a `numpy.ndarray`, a `COMPSsArray` is mutable and does not return views. As such, the following methods act in-place and return nothing:
    - `reshape`
    - `transpose`
    """

    # pylint: disable=protected-access
    @__init__.register
    def __init__(self, arr: SupportsArray):
        "Constructor for generic arrays."
        self.data = np.array(arr)
        self.__shape = arr.shape
        self.__dtype = arr.dtype

    @__init__.register
    def __init__(self, arr: np.generic):
        "Constructor for scalars."
        self.data = arr
        self.__shape = ()
        self.__dtype = arr.dtype

    @__init__.register
    def __init__(self, arr: COMPSsFuture, shape, dtype):
        "Constructor for future result of COMPSs tasks."
        self.data = arr
        self.__shape = shape
        self.__dtype = dtype

    @multimethod
    def __init__(self, arr, **kwargs):
        if not hasattr(arr, "__array__"):
            raise TypeError(f"You must provide a numpy.ndarray or a COMPSs future to a numpy.ndarray, but a {type(arr)} was provided")

        self.data = arr
        self.__shape = arr.shape if hasattr(arr, "shape") else kwargs["shape"]
        self.__dtype = arr.dtype if hasattr(arr, "dtype") else kwargs["dtype"]

        assert isinstance(self.dtype, (np.dtype, type))
        self.__dtype = np.dtype(self.__dtype)

    @classmethod
    def __infer_type_parameter__(cls, *args, **kwargs) -> Type:
        "Returns the parameter type."
        # TODO support more parametric types? cupy?
        return np.ndarray

    def __del__(self):
        compss_delete_object(self.data)

    def __getitem__(self, idx) -> COMPSsFuture:
        return compss_wait_on(task.getitem(self.data, idx))

    def __setitem__(self, key, value):
        task.setitem(self.data, key, value)

    @property
    def ref(self):
        return self.data

    @property
    def shape(self) -> Tuple[int]:
        return self.__shape

    @todo
    @shape.setter
    def _(self, value: Tuple[int]):
        raise NotImplementedError("reshape call from shape.setter not implemented")

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def ndim(self) -> int:
        return len(self.__shape)

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    def __deepcopy__(self, memo):
        ref = task.copy(self.data)
        return COMPSsArray(ref, shape=self.shape, dtype=self.dtype)

    def __array__(self) -> np.ndarray:
        return compss_wait_on(self.data)

    def __array_priority__(self) -> int:
        # NOTE higher priority than numpy.ndarray, lower than DataClayArray
        return 1

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if ufunc.nin > 2:
            return NotImplemented

        # get COMPSs reference if COMPSsArray
        inputs = [arg.data if isinstance(arg, self.__class__) else arg for arg in inputs]

        inplace = False
        if "out" in kwargs and kwargs["out"] == (self,):
            inplace = True
            kwargs["out"] = (self.data,)

        # 'at' operates in-place
        if method == "at":
            if not np.can_cast(inputs[1], inputs[0], casting="safe"):
                return NotImplemented
            task.ioperate(ufunc, *inputs, **kwargs)

        # '__call__', 'outer'
        elif method in "__call__":
            if inplace:
                types = [i.dtype if hasattr(i, "dtype") else i for i in inputs]
                if not np.can_cast(types[1], types[0], casting="safe"):
                    return NotImplemented
                task.ioperate(ufunc, *inputs, **kwargs)
                return self
            else:
                ref = task.operate(ufunc, *inputs, **kwargs)
                dtype = np.result_type(*(i.dtype if hasattr(i, "dtype") else i for i in inputs))
                return COMPSsArray(ref, shape=self.shape, dtype=dtype)

        elif method == "outer":
            if inplace:
                return NotImplemented
            else:
                ref = task.operate(ufunc, *inputs, **kwargs)
                shape = functools.reduce(tuple.__add__, (i.shape for i in inputs))
                dtype = np.result_type(*(i.dtype for i in inputs))
                return COMPSsArray(ref, shape=shape, dtype=dtype)

        # 'reduce', 'accumulate', 'reduceat' not supported yet
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        # use autoray for dispatching
        # TODO use wrap for specialization?
        f = None
        with suppress(AttributeError):
            f = autoray.get_lib_fn("rosnet", func.__name__)

            if f is None:
                f = autoray.get_lib_fn("rosnet.COMPSsArray", func.__name__)

            if f is None:
                f = autoray.get_lib_fn("rosnet.COMPSsArray.random", func.__name__)

        # multiple dispatch specialization takes place here with multimethod
        return f(*args, **kwargs) if f else NotImplemented


@iface.to_numpy.register
def to_numpy(arr: BlockArray[COMPSsArray]):
    blocks = np.empty_like(self.data, dtype=object)
    it = np.nditer(
        self.data,
        flags=["refs_ok", "multi_index"],
        op_flags=["readonly"],
        op_axes=[tuple(range(self.ndim))],
    )
    with it:
        for x in it:
            blocks[it.multi_index] = compss_wait_on(np.array(x[()]))
    return np.block(blocks)


@implements(np.zeros, ext="COMPSsArray")
def zeros(shape, dtype=None, order="C") -> COMPSsArray:
    return COMPSsArray.full(shape, 0, dtype=dtype, order=order)


@implements(np.ones, ext="COMPSsArray")
def ones(shape, dtype=None, order="C") -> COMPSsArray:
    return COMPSsArray.full(shape, 1, dtype=dtype, order=order)


@implements(np.full, ext="COMPSsArray")
def full(shape, fill_value, dtype=None, order="C") -> COMPSsArray:
    ref = task.full(shape, fill_value, dtype=dtype, order=order)
    return COMPSsArray(ref, shape=shape, dtype=dtype or np.dtype(type(fill_value)))


@iface.zeros_like.register
def zeros_like(a: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@iface.ones_like.register
def ones_like(a: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@iface.full_like.register
def full_like(a: COMPSsArray, fill_value, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@iface.empty_like.register
def empty_like(prototype: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@iface.reshape.register
def reshape(a: COMPSsArray, shape, order="F", inplace=True):
    # pylint: disable=protected-access
    a = a if inplace else deepcopy(a)

    # TODO support order?
    task.reshape(a.data, shape)
    a.shape = shape
    return a


@iface.transpose.register
def transpose(a: COMPSsArray, axes=None, inplace=True):
    # pylint: disable=protected-access
    if not isunique(axes):
        raise ValueError("'axes' must be a unique list: %s" % axes)

    a = a if inplace else deepcopy(a)

    task.transpose(a.data, axes)
    a.__shape = tuple(a.__shape[i] for i in axes)

    return a


@todo
@iface.stack.register
def stack(arrays: Sequence[COMPSsArray], axis=0, out=None) -> COMPSsArray:
    pass


@todo
@iface.split.register
def split(array: COMPSsArray, indices_or_sections, axis=0) -> list[COMPSsArray]:
    pass


@iface.tensordot.register(COMPSsArray, SupportsArray)
@iface.tensordot.register(SupportsArray, COMPSsArray)
def tensordot(a: Union[COMPSsArray, SupportsArray], b: Union[COMPSsArray, SupportsArray], axes):
    a = a if isinstance(a, COMPSsArray) else COMPSsArray(a)
    b = b if isinstance(b, COMPSsArray) else COMPSsArray(b)
    return tensordot.invoke(COMPSsArray, COMPSsArray)(a, b, axes)


@iface.tensordot.register
def tensordot(a: COMPSsArray, b: COMPSsArray, axes) -> COMPSsArray:
    dtype = np.result_type(a.dtype, b.dtype)
    shape = result_shape(a.shape, b.shape, axes)

    ref = task.tensordot.tensordot(a._data, b.data, axes)
    return COMPSsArray(ref, shape=shape, dtype=dtype)


@iface.tensordot.register
def tensordot(a: Sequence[COMPSsArray], b: Sequence[COMPSsArray], axes, method="sequential") -> COMPSsArray:
    dtype = np.result_type(a.dtype, b.dtype)
    shape = result_shape(a[0].shape, b[0].shape, axes)

    # TODO refactor method names
    if method == "sequential":
        a = [i.ref for i in a]
        b = [i.ref for i in b]
        ref = task.tensordot.sequential(a, b, axes)
    elif method == "commutative":
        ref = MaybeArray()
        for ia, ib in zip(a, b):
            task.tensordot.commutative(ref, ia, ib, axes)
    elif method == "commutative-but-first":
        ref = task.tensordot.tensordot(a[0].ref, b[0].ref, axes)
        for ia, ib in zip(a[1:], b[1:]):
            task.tensordot.commutative(ref, ia, ib, axes)
    else:
        raise ValueError("invalid method")
    return COMPSsArray(ref, shape=shape, dtype=dtype)


# @implements(np.block, COMPSsArray)
# def __compss_block(arrays):
#     return np.block(compss_wait_on([a.data for a in arrays]))


@implements("random.rand", ext="COMPSsArray")
def rand(shape):
    # TODO support inner as in BlockArray
    dtype = np.dtype(np.float64)
    return COMPSsArray(task.init.rand(blockshape), shape=shape, dtype=dtype)
