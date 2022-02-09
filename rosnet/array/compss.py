from typing import Tuple, Type, Optional, Sequence, List
import functools
from contextlib import suppress
from copy import deepcopy
from math import prod
import numpy as np
from plum import dispatch
import autoray
from pycompss.runtime.management.classes import Future as COMPSsFuture
from pycompss.api.api import compss_delete_object, compss_wait_on
from rosnet.helper.macros import todo, implements
from rosnet.helper.math import result_shape
from rosnet.helper.typing import Array, SupportsArray
from rosnet import task, tuning


# TODO special variation for in-place functions? keep np.reshape/transpose/... as non-modifying -> create new COMPSsArray/BlockArray
# TODO support more properties of ndarray
class COMPSsArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Reference to a `numpy.ndarray` managed by COMPSs.

    Unlike a `numpy.ndarray`, a `COMPSsArray` is mutable and does not return views. As such, the following methods act in-place and return nothing:
    - `reshape`
    - `transpose`
    """

    # pylint: disable=protected-access
    @dispatch
    def __init__(self, arr: SupportsArray):
        "Constructor for generic arrays."
        self._ref = np.array(arr)
        self.__shape = arr.shape
        self.__dtype = arr.dtype

    @dispatch(precedence=1)
    def __init__(self, arr: np.generic):
        "Constructor for scalars."
        self._ref = arr
        self.__shape = ()
        self.__dtype = arr.dtype

    @dispatch
    def __init__(self, arr: COMPSsFuture, shape, dtype):
        "Constructor for future result of COMPSs tasks."
        self._ref = arr
        self.__shape = shape
        self.__dtype = dtype

    @dispatch
    def __init__(self, arr, **kwargs):
        if not hasattr(arr, "__array__"):
            raise TypeError(f"You must provide a numpy.ndarray or a COMPSs future to a numpy.ndarray, but a {type(arr)} was provided")

        self._ref = arr
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
        ref = task.copy(self._ref)
        return COMPSsArray(ref, shape=self.shape, dtype=self.dtype)

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
        # use autoray for dispatching
        # TODO use wrap for specialization?
        f = None
        with suppress(AttributeError):
            f = autoray.get_lib_fn("rosnet", func.__name__)

            if f is None:
                f = autoray.get_lib_fn("rosnet.COMPSsArray", func.__name__)

            if f is None:
                f = autoray.get_lib_fn("rosnet.COMPSsArray.random", func.__name__)

        # multiple dispatch specialization takes place here with plum
        return f(*args, **kwargs) if f else NotImplemented


# TODO waiting to https://github.com/wesselb/plum/issues/37
@dispatch
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


@dispatch
def zeros_like(a: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@dispatch
def ones_like(a: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@dispatch
def full_like(a: COMPSsArray, fill_value, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@dispatch
def empty_like(prototype: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@dispatch
def reshape(a: COMPSsArray, shape, order="F", inplace=True):
    # pylint: disable=protected-access
    a = a if inplace else deepcopy(a)

    # TODO support order?
    task.reshape(a._ref, shape)
    a.shape = shape
    return a


@dispatch
def transpose(a: COMPSsArray, axes=None, inplace=True):
    # pylint: disable=protected-access
    if not isunique(axes):
        raise ValueError("'axes' must be a unique list: %s" % axes)

    a = a if inplace else deepcopy(a)

    task.transpose(a._ref, axes)
    a.__shape = tuple(a.__shape[i] for i in axes)

    return a


@todo
@dispatch
def stack(arrays: Sequence[COMPSsArray], axis=0, out=None) -> COMPSsArray:
    pass


@todo
@dispatch
def split(array: COMPSsArray, indices_or_sections, axis=0) -> List[COMPSsArray]:
    pass


@dispatch.multi((COMPSsArray, SupportsArray), (SupportsArray, COMPSsArray))
def tensordot(a: Union[COMPSsArray, SupportsArray], b: Union[COMPSsArray, SupportsArray], axes):
    a = a if isinstance(a, COMPSsArray) else COMPSsArray(a)
    b = b if isinstance(b, COMPSsArray) else COMPSsArray(b)
    return tensordot.invoke(COMPSsArray, COMPSsArray)(a, b, axes)


@dispatch(precedence=1)
def tensordot(a: COMPSsArray, b: COMPSsArray, axes) -> COMPSsArray:
    # pylint: disable=protected-access
    # TODO assertions

    dtype = np.result_type(a.dtype, b.dtype)
    shape = result_shape(a.shape, b.shape, axes)

    ref = task.tensordot.tensordot(a._ref, b._ref, axes)
    return COMPSsArray(ref, shape=shape, dtype=dtype)


@dispatch(precedence=1)
def tensordot(a: BlockArray[COMPSsArray], b: BlockArray[COMPSsArray], axes):
    # pylint: disable=protected-access
    # TODO assertions
    # TODO selection of implementation based on input arrays

    # iterators
    outer_axes = [list(set(range(i.ndim)) - set(ax)) for ax, i in zip(axes, (a, b))]
    outer_iter_a, inner_iter_a = np.nested_iters(
        a.data,
        [outer_axes[0], axes[0]],
        op_flags=["readonly"],
        flags=["multi_index", "refs_ok"],
    )
    outer_iter_b, inner_iter_b = np.nested_iters(
        b.data,
        [outer_axes[1], axes[1]],
        op_flags=["readonly"],
        flags=["multi_index", "refs_ok"],
    )

    grid = np.empty(outer_iter_a.shape + outer_iter_b.shape, dtype=COMPSsArray)
    dtype = np.result_type(a.dtype, b.dtype)
    blockshape = result_shape(a.blockshape, b.blockshape, axes)

    # estimate number of cores per task for dynamic parallelism
    impl, ncores = tuning.tensordot(a, b, axes)

    # required memory per task
    # pylint: disable=no-member
    # memory = a.blocknbytes + b.blocknbytes + prod(blockshape) * dtype.itemsize
    # pylint: enable=no-member

    with tuning.allocate(ncores=ncores):  # , memory=memory):
        for _ in outer_iter_a:  # outer_i_a
            for _ in outer_iter_b:  # outer_i_b
                idx = outer_iter_a.multi_index + outer_iter_b.multi_index

                # call chosen implementation
                blocks_a = list(
                    map(
                        lambda x: a.data[x],
                        (
                            join_idx(
                                outer_iter_a.multi_index,
                                inner_iter_a.multi_index,
                                axes[0],
                            )
                            for _ in inner_iter_a
                        ),
                    )
                )
                blocks_b = list(
                    map(
                        lambda x: b.data[x],
                        (
                            join_idx(
                                outer_iter_b.multi_index,
                                inner_iter_b.multi_index,
                                axes[1],
                            )
                            for _ in inner_iter_b
                        ),
                    )
                )

                grid[idx] = impl(blocks_a, blocks_b, axes)

                # reset inner block iterators
                inner_iter_a.reset()
                inner_iter_b.reset()
            outer_iter_b.reset()

    return BlockArray(grid)


# @implements(np.block, COMPSsArray)
# def __compss_block(arrays):
#     return np.block(compss_wait_on([a._ref for a in arrays]))


@implements("random.rand", ext="COMPSsArray")
def rand(shape):
    # TODO support inner as in BlockArray
    dtype = np.dtype(np.float64)
    return COMPSsArray(task.init.rand(blockshape), shape=shape, dtype=dtype)
