from typing import Tuple, Type, List, Sequence, Optional
from math import prod
from copy import deepcopy
from functools import reduce
import numpy as np
from plum import dispatch, parametric
import autoray
from rosnet.helper.math import (
    isunique,
    space,
    result_shape,
    join_idx,
    recurse,
)
from rosnet.helper.macros import todo, implements
from rosnet.helper.typing import Array, SupportsArray


@parametric
class BlockArray(np.lib.mixins.NDArrayOperatorsMixin):
    """A n-dimensional array divided in blocks.

    Implementation notes
    --------------------
    - All blocks are expected to have the same type and `dtype`.
    - All blocks are expected to be equally sized.
    """

    data: np.ndarray = None

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], List):
            self.__init_with_list__(*args, **kwargs)
        elif isinstance(args[0], np.ndarray):
            self.__init_with_array__(*args, **kwargs)
        else:
            raise ValueError("invalid constructor")

    def __init_with_list__(self, blocks: List, grid: Optional[Sequence[int]] = None):
        """Constructor.

        Arguments
        ---------
        - blocks: List[Array]. Nested list of arrays.
        """
        if grid is None:
            grid = [len(blocks)]
            for i in recurse(blocks):
                grid += [len(i)]

            grid = tuple(grid)

        self.data = np.empty(grid, dtype=object)

        it = np.nditer(self.data, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

        with it:
            for block in it:
                # case for nested list of arrays
                if isinstance(blocks[0], List):
                    data = blocks
                    for i in it.multi_index:
                        data = data[i]
                # case for list of arrays
                else:
                    data = blocks[it.iterindex]

                if isinstance(data, Array):
                    block[()] = data
                elif isinstance(data, SupportsArray):
                    block[()] = np.array(data)
                else:
                    raise ValueError("blocks must provide an array-like interface")

        if grid:
            self.data = self.data.reshape(grid)

    def __init_with_array__(self, arr):
        """Constructor."""

        level = 0
        for i in recurse(arr):
            level += 1

        if level:
            self.data = arr.copy()
        else:
            self.data = np.empty(tuple(1 for _ in arr.shape), dtype=object)
            self.data.flat[0] = arr

    @classmethod
    def __infer_type_parameter__(cls, *args, **kwargs) -> type:
        "Returns the type parameter."

        x = args[0]
        try:
            while True:
                x = x.flat[0] if isinstance(x, np.ndarray) else x[0]
        except:
            return type(x)

    def __str__(self):
        return "BlockArray(shape=%r, grid=%r, blockshape=%r, dtype=%r)" % (
            self.shape,
            self.grid,
            self.blockshape,
            self.dtype,
        )

    @dispatch
    def __getitem__(self, index: List[int]):
        # TODO advanced indexing
        if len(index) != self.ndim:
            raise IndexError(f"Invalid indexing: index={index}")

        gid = [i // s for i, s in zip(key, self.blockshape)]
        bid = [i % s for i, s in zip(key, self.blockshape)]

        return self.data[gid][bid]

    @dispatch
    def __setitem__(self, key: List[int], value):
        # TODO advanced indexing
        if len(key) != self.ndim:
            raise IndexError(f"Invalid indexing: key={key}")

        gid = [i // s for i, s in zip(key, self.blockshape)]
        bid = [i % s for i, s in zip(key, self.blockshape)]

        self.data[gid][bid] = value

    @property
    def shape(self) -> Tuple[int]:
        return tuple(np.multiply(self.grid, self.blockshape))

    @property
    def blockshape(self) -> Tuple[int]:
        return self.data.flat[0].shape

    @property
    def nblock(self) -> int:
        return self.data.size

    @property
    def grid(self) -> Tuple[int]:
        return self.data.shape

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
        return self.data.flat[0].dtype

    def __deepcopy__(self, memo):
        grid = np.empty_like(self.data)
        for i in self.nblock:
            grid.flat[i] = deepcopy(self.data.flat[i])

        return BlockArray(grid)

    def __array__(self) -> np.ndarray:
        "Returns a numpy.ndarray. Uses class-parametric specialization with plum."
        return to_numpy(self)

    @todo
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        pass

    def __array_function__(self, func, types, args, kwargs):
        # use autoray for dispatching
        # TODO use wrap for specialization?
        f = None
        with suppress(AttributeError):
            f = autoray.get_lib_fn("rosnet", func.__name__)

            if f is None:
                f = autoray.get_lib_fn("rosnet.BlockArray", func.__name__)

            if f is None:
                f = autoray.get_lib_fn("rosnet.BlockArray.random", func.__name__)

        # multiple dispatch specialization takes place here with plum
        return f(*args, **kwargs) if f else NotImplemented


@dispatch
def to_numpy(arr: BlockArray) -> np.ndarray:
    return np.block(arr.data.tolist())


@implements(np.zeros, ext="BlockArray")
def zeros(shape, dtype=None, order="C", blockshape=None, inner="numpy") -> BlockArray:
    return autoray.do("full", shape, 0, dtype=dtype, order=order, blockshape=blockshape, backend="rosnet.BlockArray")


@implements(np.ones, ext="BlockArray")
def ones(shape, dtype=None, order="C", blockshape=None, inner="numpy") -> BlockArray:
    return autoray.do("full", shape, 1, dtype=dtype, order=order, blockshape=blockshape, backend="rosnet.BlockArray")


@implements(np.full, ext="BlockArray")
def full(shape, fill_value, dtype=None, order="C", blockshape=None, inner="numpy") -> BlockArray:
    dtype = dtype or np.dtype(type(value))
    blocks = np.empty_like(blocks, dtype=object)
    it = np.nditer(blocks, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

    with it:
        for block in it:
            block[()] = autoray.do("full", blockshape, value, dtype=dtype, order=order, backend=inner)

    return BlockArray(blocks)


@dispatch
def zeros_like(a: BlockArray, dtype=None, order="K", subok=True, shape=None) -> BlockArray:
    pass


@dispatch
def ones_like(a: BlockArray, dtype=None, order="K", subok=True, shape=None) -> BlockArray:
    pass


@dispatch
def full_like(a: BlockArray, fill_value, dtype=None, order="K", subok=True, shape=None) -> BlockArray:
    pass


@dispatch
def empty_like(prototype: BlockArray, dtype=None, order="K", subok=True, shape=None) -> BlockArray:
    pass


@dispatch
def reshape(a: BlockArray, shape, order="F", inplace=True):
    a = a if inplace else deepcopy(a)

    # TODO reshape blocks? or blockshape? for now, blockshape
    for i, block in enumerate(a.data.flat):
        a.data.flat[i] = autoray.do("reshape", block, shape, order=order)

    a.__blockshape = shape

    return a


@dispatch
def transpose(a: BlockArray, axes=None, inplace=True):
    # pylint: disable=protected-access
    if not isunique(axes):
        raise ValueError("'axes' must be a unique list: %s" % axes)

    a = a if inplace else deepcopy(a)

    for i, block in enumerate(a.data.flat):
        a.data.flat[i] = autoray.do("transpose", block, axes)

    a.data = np.transpose(a.data, axes)
    a.__blockshape = tuple(a.__blockshape[i] for i in axes)

    return a


@dispatch
def tensordot(a: BlockArray, b: BlockArray, axes):
    # pylint: disable=protected-access
    # TODO assertions
    # TODO generic BlockArray implementation and call to specialized tensordot of lists of blocks?

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

    grid = np.empty(outer_iter_a.shape + outer_iter_b.shape)
    dtype = np.result_type(a.dtype, b.dtype)
    blockshape = result_shape(a.blockshape, b.blockshape, axes)

    for _ in outer_iter_a:
        for _ in outer_iter_b:
            idx = outer_iter_a.multi_index + outer_iter_b.multi_index

            bid_a = (
                join_idx(
                    outer_iter_a.multi_index,
                    inner_iter_a.multi_index,
                    axes[0],
                )
                for _ in inner_iter_a
            )
            bid_b = (
                join_idx(
                    outer_iter_b.multi_index,
                    inner_iter_b.multi_index,
                    axes[1],
                )
                for _ in inner_iter_b
            )

            grid[idx] = np.sum(np.tensordot(a.data[ba], b.data[bb], axes) for ba, bb in zip(bid_a, bid_b))

            # reset inner block iterators
            inner_iter_a.reset()
            inner_iter_b.reset()
        outer_iter_b.reset()

    return BlockArray(grid)


# @todo
# @implements(np.array, BlockArray)
# def array(
#     arr: np.ndarray,
#     dtype=None,
#     blockshape=None,
#     copy=True,
#     order="K",
#     subok=False,
#     ndmin=0,
# ):
#     """
#     `dtype`, `copy`, `order`, `subok` and `ndmin` ignored at the moment.
#     """
#     shape = arr.shape
#     blockshape = shape if blockshape is None else blockshape
#     grid = [s // bs for s, bs in zip(shape, blockshape)]

#     blocks = []
#     for bidx in space(grid):
#         idx_begin = [x * y for x, y in zip(blockshape, bidx)]
#         idx_end = [x + y for x, y in zip(blockshape, idx_begin)]
#         idx = tuple(slice(b, e) for b, e in zip(idx_begin, idx_end))
#         block = arr[idx]
#         blocks.append(COMPSsArray(block, shape=blockshape, dtype=arr.dtype))

#     blocks = ndarray_from_list(blocks, grid)
#     return BlockArray(blocks)


@implements("random.rand", ext="BlockArray")
def rand(shape, blockshape=None, inner="numpy"):
    blockshape = shape if blockshape is None else blockshape
    blocks = np.empty_like(blocks, dtype=object)
    it = np.nditer(blocks, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

    with it:
        for block in it:
            block[()] = autoray.do("random.rand", blockshape, backend=inner)

    return BlockArray(blocks)
