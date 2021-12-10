from typing import Tuple, Type
import functools
import numpy as np
from pycompss.runtime.management.classes import Future as COMPSsFuture
from pycompss.api.api import compss_delete_object, compss_wait_on
from rosnet.helper.macros import todo, implements, numpy_dispatcher
from rosnet.helper.math import prod, result_shape
from rosnet import task


# TODO special variation for in-place functions? keep np.reshape/transpose/... as non-modifying -> create new COMPSsArray/BlockArray
# TODO support more properties of ndarray
class COMPSsArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Reference to a `numpy.ndarray` managed by COMPSs.

    Unlike a `numpy.ndarray`, a `COMPSsArray` is mutable and does not return views. As such, the following methods act in-place and return nothing:
    - `reshape`
    - `transpose`
    """

    # pylint: disable=protected-access

    def __init__(self, *args, **kwargs):
        self._ref = args[0]
        if issubclass(args[0].__class__, np.ndarray) or isinstance(args[0], np.ndarray):
            self.__shape = args[0].shape
            self.__dtype = args[0].dtype
        elif isinstance(args[0], COMPSsFuture):
            self.__shape = kwargs["shape"]
            self.__dtype = kwargs["dtype"]
        elif isinstance(args[0], np.generic):
            self.__shape = ()
            self.__dtype = args[0].dtype
        elif hasattr(args[0], "__array__"):
            self.__shape = args[0].shape if hasattr(args[0], "shape") else kwargs["shape"]
            self.__dtype = args[0].dtype if hasattr(args[0], "dtype") else kwargs["dtype"]
        else:
            raise TypeError("You must provide a np.ndarray or a COMPSs Future to a np.ndarray, but a %r was provided" % type(args[0]))

        if isinstance(self.dtype, type):
            self.__dtype = np.dtype(self.__dtype)

        assert isinstance(self.dtype, np.dtype)

    def __del__(self):
        compss_delete_object(self._ref)

    def __getitem__(self, idx) -> COMPSsFuture:
        return compss_wait_on(task.getitem(self._ref, idx))

    def __setitem__(self, key, value):
        task.setitem(self._ref, key, value)

    @property
    def shape(self) -> Tuple[int]:
        return self.__shape

    @todo
    @shape.setter
    def _(self, value: Tuple[int]):
        # self.reshape(value)
        pass

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
        ref = task.copy(self._ref)
        return COMPSsArray(ref, shape=self.shape, dtype=self.dtype)

    # def sync(self):
    #     self._ref = compss_wait_on(self._ref)

    def __array__(self) -> np.ndarray:
        return compss_wait_on(self._ref)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if ufunc.nin > 2:
            return NotImplemented

        # get COMPSs reference if COMPSsArray
        inputs = [arg._ref if isinstance(arg, self.__class__) else arg for arg in inputs]

        inplace = False
        if "out" in kwargs and kwargs["out"] == (self,):
            inplace = True
            kwargs["out"] = (self._ref,)

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
        if func not in numpy_dispatcher[self.__class__]:
            print("not in dispatcher")
            return NotImplemented
        if not all(t == self.__class__ for t in types):
            print("bad type")
            # TODO create COMPSsArray if np.ndarray implementation
            return NotImplemented

        return numpy_dispatcher[self.__class__][func](*args, **kwargs)


@implements(np.zeros, COMPSsArray)
def __compss_zeros(shape, dtype=None, order="C"):
    return np.full(shape, 0, dtype=dtype, order=order)


@implements(np.ones, COMPSsArray)
def __compss_ones(shape, dtype=None, order="C"):
    return np.full(shape, 1, dtype=dtype, order=order)


@implements(np.full, COMPSsArray)
def __compss_full(shape, fill_value, dtype=None, order="C"):
    ref = task.full(shape, fill_value, dtype=dtype, order=order)
    return COMPSsArray(ref, shape=shape, dtype=dtype or np.dtype(type(fill_value)))


# numpy: chainging array shape
@implements(np.reshape, COMPSsArray)
def __compss_reshape(a: COMPSsArray, newshape: Tuple[int], order="F"):
    # pylint: disable=protected-access
    # TODO support order?
    # TODO should return something?
    task.reshape(a._ref, newshape)


# numpy: transpose-like operations
@implements(np.transpose, COMPSsArray)
def __compss_transpose(a: COMPSsArray, axes: Tuple[int] = None):
    # pylint: disable=protected-access
    # TODO should return something?
    task.transpose(a._ref, axes)


# numpy: joining arrays
@todo
@implements(np.stack, COMPSsArray)
def __compss_stack(self):
    pass


# numpy: splitting arrays
@todo
@implements(np.split, COMPSsArray)
def __compss_split(self, indices_or_sections, axis=0):
    pass


@implements(np.tensordot, COMPSsArray)
def __compss_tensordot(a: COMPSsArray, b: COMPSsArray, axes):
    # pylint: disable=protected-access
    # TODO assertions

    # only support operating against COMPSsArray
    if not all(isinstance(i, COMPSsArray) for i in (a, b)):
        return NotImplemented

    dtype = np.result_type(a.dtype, b.dtype)
    shape = result_shape(a.shape, b.shape, axes)

    ref = task.tensordot.tensordot(a._ref, b._ref, axes)
    return COMPSsArray(ref, shape=shape, dtype=dtype)


# @implements(np.block, COMPSsArray)
# def __compss_block(arrays):
#     return np.block(compss_wait_on([a._ref for a in arrays]))
