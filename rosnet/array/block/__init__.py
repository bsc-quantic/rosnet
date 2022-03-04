import sys
from typing import Tuple, Type, Sequence, Optional, Generic, TypeVar
from math import prod
from copy import deepcopy
import numpy as np
from multimethod import multimethod
import autoray
from rosnet.core.math import (
    isunique,
    space,
    result_shape,
    join_idx,
    measure_shape,
    nest_level,
)
from rosnet.core.macros import todo
from rosnet.core.interface import Array, ArrayConvertable
from rosnet.core.mixin import ArrayFunctionMixin
from rosnet import dispatch as dispatcher

T = TypeVar("T", Array, np.ndarray, covariant=True)


class BlockArray(np.lib.mixins.NDArrayOperatorsMixin, ArrayFunctionMixin, Generic[T]):
    """A n-dimensional array divided in blocks.

    Implementation notes
    --------------------
    - All blocks are expected to have the same type and `dtype`.
    - All blocks are expected to be equally sized.
    - Automatic parametric type detection works only on Python 3.9 or later. On earlier versions, you must
    """

    data: np.ndarray = None

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], list):
            self.__init_with_list__(*args, **kwargs)
        elif isinstance(args[0], ArrayConvertable):
            self.__init_with_array__(*args, **kwargs)
        else:
            raise ValueError("invalid constructor")

        # NOTE multimethod checks obj.__orig_class__ for parametric multiple-dispatch, which is a instance of GenericAlias that stores the original class and the parametric type.
        # GenericAlias is standardized in Python 3.9, which is needed for automatic parameter type detection.
        # On earlier versions, GenericAlias is a implementation detail (i.e. _GenericAlias).
        if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
            from types import GenericAlias

            self.__orig_class__ = GenericAlias(self.__class__, self.data.flat[0].__class__)
        else:
            # TODO debug message that automatic parametric type detection does not work
            pass

    def __init_with_list__(self, blocks: list, grid: Optional[Sequence[int]] = None):
        """Constructor.

        Arguments
        ---------
        - blocks: list[Array]. Nested list of arrays.
        """
        if grid is None:
            grid = measure_shape(blocks)

        self.data = np.empty(grid, dtype=object)

        it = np.nditer(self.data, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

        with it:
            for block in it:
                # case for nested list of arrays
                if isinstance(blocks[0], list):
                    data = blocks
                    for i in it.multi_index:
                        data = data[i]
                # case for list of arrays
                else:
                    data = blocks[it.iterindex]

                if data.ndim != self.data.ndim:
                    raise ValueError("blocks and grid should have same ndim. append single-dimensions (1) where needed")

                if isinstance(data, Array):
                    block[()] = data
                elif isinstance(data, ArrayConvertable):
                    block[()] = np.array(data)
                else:
                    raise ValueError("blocks must provide an array-like interface")

    def __init_with_array__(self, arr):
        """Constructor."""

        if nest_level(arr):
            self.data = arr
        else:
            self.data = np.empty(tuple(1 for _ in arr.shape), dtype=object)
            self.data.flat[0] = arr

    def __class_getitem__(cls, item):
        if not issubclass(item, Array):
            raise TypeError(f"'BlockArray[{item.__name__}]' is not a valid type as '{item.__name__}' does not fulfill the Array protocol")

        return super().__class_getitem__(item)

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

    @multimethod
    def __getitem__(self, index: Sequence[int]):
        # TODO advanced indexing
        if len(index) != self.ndim:
            raise IndexError(f"Invalid indexing: index={index}")

        gid = [i // s for i, s in zip(key, self.blockshape)]
        bid = [i % s for i, s in zip(key, self.blockshape)]

        return self.data[gid][bid]

    @multimethod
    def __setitem__(self, key: Sequence[int], value):
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
        "Returns a numpy.ndarray. Uses class-parametric specialization with multimethod."
        return dispatcher.to_numpy(self)

    @todo
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        pass


@dispatcher.to_numpy.register
def to_numpy(arr: BlockArray) -> np.ndarray:
    return np.block(arr.data.tolist())


def zeros(shape, dtype=None, order="C", blockshape=None, inner="numpy") -> BlockArray:
    return full(shape, 0, dtype=dtype, order=order, blockshape=blockshape, inner=inner)


def ones(shape, dtype=None, order="C", blockshape=None, inner="numpy") -> BlockArray:
    return full(shape, 1, dtype=dtype, order=order, blockshape=blockshape, inner=inner)


def full(shape, fill_value, dtype=None, order="C", blockshape=None, inner="numpy") -> BlockArray:
    dtype = dtype or np.dtype(type(fill_value))
    blockshape = blockshape or shape
    grid = tuple(s // bs for s, bs in zip(shape, blockshape))

    blocks = np.empty(grid, dtype=object)
    it = np.nditer(blocks, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

    with it:
        for block in it:
            block[()] = autoray.do("full", blockshape, fill_value, dtype=dtype, order=order, like=inner)

    return BlockArray(blocks)


@dispatcher.zeros_like.register
def zeros_like(a: BlockArray, dtype=None, order="K", subok=True, shape=None) -> BlockArray:
    pass


@dispatcher.ones_like.register
def ones_like(a: BlockArray, dtype=None, order="K", subok=True, shape=None) -> BlockArray:
    pass


@dispatcher.full_like.register
def full_like(a: BlockArray, fill_value, dtype=None, order="K", subok=True, shape=None) -> BlockArray:
    pass


@dispatcher.empty_like.register
def empty_like(prototype: BlockArray, dtype=None, order="K", subok=True, shape=None) -> BlockArray:
    pass


@dispatcher.reshape.register
def reshape(a: BlockArray, shape, order="F", inplace=True):
    a = a if inplace else deepcopy(a)

    # TODO reshape blocks? or blockshape? for now, blockshape
    for i, block in enumerate(a.data.flat):
        a.data.flat[i] = autoray.do("reshape", block, shape, order=order)

    a.__blockshape = shape

    return a


@dispatcher.transpose.register
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


@dispatcher.tensordot.register
def tensordot(a: Sequence[Array], b: Sequence[Array], axes) -> Array:
    return sum(np.tensordot(ai, bi, axes) for ai, bi in zip(a, b))


@dispatcher.tensordot.register
def tensordot(a: BlockArray, b: BlockArray, axes):
    # pylint: disable=protected-access
    # TODO assertions

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

    grid = np.empty(outer_iter_a.shape + outer_iter_b.shape, dtype=object)

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

            # call specialized tensordot routine
            grid[idx] = tensordot([a.data[i] for i in bid_a], [b.data[i] for i in bid_b], axes)

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


def rand(shape, blockshape=None, inner="numpy"):
    blockshape = shape if blockshape is None else blockshape
    blocks = np.empty_like(blocks, dtype=object)
    it = np.nditer(blocks, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

    with it:
        for block in it:
            block[()] = autoray.do("random.rand", blockshape, like=inner)

    return BlockArray(blocks)
