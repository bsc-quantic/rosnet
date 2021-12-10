from typing import Tuple, Type
import copy
import numpy as np
from pycompss.api.api import compss_wait_on
from rosnet.array.compssarray import COMPSsArray
from rosnet.helper.math import (
    prod,
    isunique,
    ndarray_from_list,
    space,
    result_shape,
    join_idx,
)
from rosnet.helper.macros import todo, implements, numpy_dispatcher
from rosnet import task, tuning


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
            raise TypeError("expected COMPSsArray or numpy.ndarray, got " % type(args[0]))

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
        return prod(self.shape)

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def blocksize(self) -> int:
        return prod(self.blockshape)

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

    blocks = [COMPSsArray(task.full(blockshape, value, dtype), shape=blockshape, dtype=dtype) for _ in range(prod(grid))]

    blocks = ndarray_from_list(blocks, grid)
    return BlockArray(blocks, blockshape=blockshape, dtype=dtype)


def rand(shape, blockshape=None):
    blockshape = shape if blockshape is None else blockshape
    grid = tuple(s // bs for s, bs in zip(shape, blockshape))
    dtype = np.dtype(np.float64)

    blocks = [COMPSsArray(task.init.rand(blockshape), shape=blockshape, dtype=dtype) for _ in range(prod(grid))]

    blocks = ndarray_from_list(blocks, grid)
    return BlockArray(blocks, blockshape=blockshape, dtype=dtype)
