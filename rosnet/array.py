from typing import Tuple, Type
import copy
import functools
import numpy as np
import pycompss
from pycompss.api.api import compss_delete_object, compss_wait_on
from rosnet import task, utils
from rosnet.utils import (
    todo,
    isunique,
    ndarray_from_list,
    prod,
    space,
    result_shape,
    join_idx,
)
from rosnet.helper import implements, numpy_dispatcher
from rosnet import tuning


def assert_axes(axes):
    assert isinstance(axes, (list, tuple)) and all(
        isinstance(ax, (list, tuple)) for ax in axes
    )
    assert all(isunique(ax) for ax in axes)
    assert len(axes[0]) == len(axes[1])


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
        elif isinstance(args[0], pycompss.runtime.management.classes.Future):
            self.__shape = kwargs["shape"]
            self.__dtype = kwargs["dtype"]
        elif isinstance(args[0], np.generic):
            self.__shape = ()
            self.__dtype = args[0].dtype
        elif hasattr(args[0], "__array__"):
            self.__shape = (
                args[0].shape if hasattr(args[0], "shape") else kwargs["shape"]
            )
            self.__dtype = (
                args[0].dtype if hasattr(args[0], "dtype") else kwargs["dtype"]
            )
        else:
            raise TypeError(
                "You must provide a np.ndarray or a COMPSs Future to a np.ndarray, but a %r was provided"
                % type(args[0])
            )

        if isinstance(self.dtype, type):
            self.__dtype = np.dtype(self.__dtype)

        assert isinstance(self.dtype, np.dtype)

    def __del__(self):
        compss_delete_object(self._ref)

    def __getitem__(self, idx):
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
        return utils.prod(self.shape)

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

    # numpy-compatibility
    def __array__(self) -> np.ndarray:
        return compss_wait_on(self._ref)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if ufunc.nin > 2:
            return NotImplemented

        # get COMPSs reference if COMPSsArray
        inputs = [
            arg._ref if isinstance(arg, self.__class__) else arg for arg in inputs
        ]

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
                dtype = np.result_type(
                    *(i.dtype if hasattr(i, "dtype") else i for i in inputs)
                )
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


class BlockArray(np.lib.mixins.NDArrayOperatorsMixin):
    """A distributed n-dimensional array divided in blocks.

    Notes
    -----------
    - All blocks have same `blockshape` and `dtype`.
    """

    _grid: np.ndarray = None
    __blockshape: Tuple = None
    __dtype: Type = None

    def __init__(self, *args, **kwargs):
        # TODO asserts
        if isinstance(args[0], COMPSsArray):
            self._grid = np.array(args[0])
            self.__blockshape = args[0].shape
            self.__dtype = args[0].dtype
        elif isinstance(args[0], np.ndarray):
            # assert asyncio.isfuture(args[0][0])
            self._grid = args[0].copy()
            self.__blockshape = kwargs["blockshape"]
            self.__dtype = kwargs.get("dtype")
        else:
            raise TypeError(
                "expected COMPSsArray or numpy.ndarray, got " % type(args[0])
            )

        if isinstance(self.dtype, type):
            self.__dtype = np.dtype(self.__dtype)

        assert isinstance(self.dtype, np.dtype)

    def __str__(self):
        return "BlockArray(shape=%r, grid=%r, blockshape=%r, dtype=%r)" % (
            self.shape,
            self.grid,
            self.blockshape,
            self.dtype,
        )

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = list(index)
        if isinstance(index, list) and all(isinstance(i, int) for i in index):
            if len(index) != self.ndim:
                raise IndexError("Invalid indexing information: index=%s" % index)

            grid_id = tuple(i // s for i, s in zip(index, self.blockshape))
            block_id = tuple(i % s for i, s in zip(index, self.blockshape))
            return compss_wait_on(task.getitem(self._grid[grid_id]._ref, block_id))

    def __setitem__(self, key, value):
        if isinstance(key, list) and isinstance(key[0], int):
            if len(key) != self.ndim:
                raise IndexError("Invalid indexing information: key=%s" % key)

            grid_id = [i // s for i, s in zip(key, self.blockshape)]
            block_id = [i % s for i, s in zip(key, self.blockshape)]
            task.setitem(self._grid[grid_id], block_id, value)

        raise IndexError("Invalid indexing information: %s" % key)

    @property
    def shape(self) -> Tuple[int]:
        return tuple(np.multiply(self._grid.shape, self.__blockshape))

    @property
    def blockshape(self) -> Tuple[int]:
        return self.__blockshape

    @property
    def nblock(self) -> int:
        return self._grid.size

    @property
    def grid(self) -> Tuple[int]:
        return self._grid.shape

    @property
    def size(self) -> int:
        return utils.prod(self.shape)

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def blocksize(self) -> int:
        return utils.prod(self.blockshape)

    @property
    def blocknbytes(self) -> int:
        return self.blocksize * self.itemsize

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    def __deepcopy__(self, memo):
        grid = np.empty_like(self._grid)
        for i in self.nblock:
            grid.flat[i] = copy.deepcopy(self._grid.flat[i])

        return BlockArray(grid, self.blockshape)

    # numpy-compatibility
    def __array__(self) -> np.ndarray:
        blocks = np.empty_like(self._grid, dtype=object)
        it = np.nditer(
            self._grid,
            flags=["refs_ok", "multi_index"],
            op_flags=["readonly"],
            op_axes=[tuple(range(self.ndim))],
        )
        with it:
            for x in it:
                blocks[it.multi_index] = compss_wait_on(x[()]._ref)
        return np.block(blocks)

    @todo
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        pass

    def __array_function__(self, func, types, args, kwargs):
        if func not in numpy_dispatcher[self.__class__]:
            print("not in dispatcher")
            return NotImplemented
        if not all(t == self.__class__ for t in types):
            print("bad type")
            # TODO create COMPSsArray if np.ndarray implementation
            return NotImplemented

        return numpy_dispatcher[self.__class__][func](*args, **kwargs)


@todo
@implements(np.reshape, BlockArray)
def __block_reshape(a, shape):
    pass


@implements(np.transpose, BlockArray)
def __block_transpose(a, axes=None):
    # pylint: disable=protected-access
    if not isunique(axes):
        raise ValueError("'axes' must be a unique list: %s" % axes)

    # transpose blocks
    for ref in a._grid.flat:
        task.transpose(ref, axes)

    # transpose grid
    a._grid = np.transpose(a._grid, axes)

    # transpose block shape
    a.__blockshape = tuple(a.__blockshape[i] for i in axes)


@implements(np.tensordot, BlockArray)
def __block_tensordot(a: BlockArray, b: BlockArray, axes):
    # pylint: disable=protected-access
    # TODO assertions
    # TODO selection of implementation based on input arrays

    # only support operating against BlockArray
    if not isinstance(b, a.__class__):
        return NotImplemented

    # iterators
    outer_axes = [list(set(range(i.ndim)) - set(ax)) for ax, i in zip(axes, (a, b))]
    outer_iter_a, inner_iter_a = np.nested_iters(
        a._grid,
        [outer_axes[0], axes[0]],
        op_flags=["readonly"],
        flags=["multi_index", "refs_ok"],
    )
    outer_iter_b, inner_iter_b = np.nested_iters(
        b._grid,
        [outer_axes[1], axes[1]],
        op_flags=["readonly"],
        flags=["multi_index", "refs_ok"],
    )

    grid = np.empty(outer_iter_a.shape + outer_iter_b.shape, dtype=COMPSsArray)
    dtype = np.result_type(a.dtype, b.dtype)
    blockshape = result_shape(a.blockshape, b.blockshape, axes)

    # estimate number of cores per task for dynamic parallelism
    impl, ncores = tuning.tuner.tensordot(a, b, axes)

    # required memory per task
    # pylint: disable=no-member
    # memory = a.blocknbytes + b.blocknbytes + utils.prod(blockshape) * dtype.itemsize
    # pylint: enable=no-member

    with tuning.resources(ncores=ncores):  # , memory=memory):
        for _ in outer_iter_a:  # outer_i_a
            for _ in outer_iter_b:  # outer_i_b
                idx = outer_iter_a.multi_index + outer_iter_b.multi_index

                # call chosen implementation
                blocks_a = list(
                    map(
                        lambda x: a._grid[x],
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
                        lambda x: b._grid[x],
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

    return BlockArray(grid, blockshape=blockshape, dtype=dtype)


@implements(np.array, BlockArray)
def array(
    arr: np.ndarray,
    dtype=None,
    blockshape=None,
    copy=True,
    order="K",
    subok=False,
    ndmin=0,
):
    """
    `dtype`, `copy`, `order`, `subok` and `ndmin` ignored at the moment.
    """
    shape = arr.shape
    blockshape = shape if blockshape is None else blockshape
    grid = [s // bs for s, bs in zip(shape, blockshape)]

    blocks = []
    for bidx in space(grid):
        idx_begin = [x * y for x, y in zip(blockshape, bidx)]
        idx_end = [x + y for x, y in zip(blockshape, idx_begin)]
        idx = tuple(slice(b, e) for b, e in zip(idx_begin, idx_end))
        block = arr[idx]
        blocks.append(COMPSsArray(block, shape=blockshape, dtype=arr.dtype))

    blocks = ndarray_from_list(blocks, grid)
    return BlockArray(blocks, blockshape=blockshape, dtype=arr.dtype)


@implements(np.zeros, BlockArray)
def zeros(shape, dtype=None, order="C", blockshape=None):
    return full(shape, 0, blockshape=blockshape, dtype=dtype)


@implements(np.ones, BlockArray)
def ones(shape, dtype=None, order="C", blockshape=None):
    return full(shape, 1, blockshape=blockshape, dtype=dtype)


@implements(np.full, BlockArray)
def full(shape, value, dtype=None, order="C", blockshape=None):
    blockshape = shape if blockshape is None else blockshape
    grid = tuple(s // bs for s, bs in zip(shape, blockshape))
    dtype = dtype or np.dtype(type(value))

    blocks = [
        COMPSsArray(task.full(blockshape, value, dtype), shape=blockshape, dtype=dtype)
        for _ in range(prod(grid))
    ]

    blocks = ndarray_from_list(blocks, grid)
    return BlockArray(blocks, blockshape=blockshape, dtype=dtype)


def rand(shape, blockshape=None):
    blockshape = shape if blockshape is None else blockshape
    grid = tuple(s // bs for s, bs in zip(shape, blockshape))
    dtype = np.dtype(np.float64)

    blocks = [
        COMPSsArray(task.init.rand(blockshape), shape=blockshape, dtype=dtype)
        for _ in range(prod(grid))
    ]

    blocks = ndarray_from_list(blocks, grid)
    return BlockArray(blocks, blockshape=blockshape, dtype=dtype)
