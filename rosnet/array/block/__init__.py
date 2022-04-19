from copy import deepcopy
from typing import Sequence

import autoray
import numpy as np
from rosnet import dispatch as dispatcher
from rosnet.core.abc import Array
from rosnet.core.util import isunique, join_idx
from .array import BlockArray


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
    grid = tuple(s // bs for s, bs in zip(shape, blockshape))
    blocks = np.empty_like(grid, dtype=object)
    it = np.nditer(blocks, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

    with it:
        for block in it:
            block[()] = autoray.do("random.rand", *blockshape, like=inner)

    return BlockArray(blocks)
