from typing import List
import functools
import itertools
import operator as op
import numpy as np

try:
    from math import prod
except ImportError:

    def prod(x):
        return functools.reduce(op.mul, x, 1)


def isunique(l: List) -> bool:
    """Checks whether all the elements in `l` are unique in `l`"""
    return len(set(l)) == len(l)


def space(s: list):
    """Generates an iterator through the Cartesian space of dimensionality `s`"""
    return itertools.product(*[range(i) for i in s])


def ispower2(n: int):
    return (n & (n - 1)) and n != 0


def isarray(o: object):
    return isinstance(o, np.ndarray) or issubclass(o.__class__, np.ndarray) or hasattr(o, "__array__")


def result_nblock(a, b, axes):
    "Returns the number of blocks of the resulting array after `tensordot`."
    return prod(prod(itertools.compress(grid, [x not in ax for x in range(len(grid))])) for ax, grid in zip(axes, (a.grid, b.grid)))


def result_shape(a, b, axes):
    "Returns the blockshape of the resulting array after `tensordot`."
    outer_axes = [tuple(set(range(len(bs))) - set(ax)) for ax, bs in zip(axes, (a, b))]
    return functools.reduce(op.add, (tuple(i[ax] for ax in outer_ax) for outer_ax, i in zip(outer_axes, (a, b))))


def join_idx(outer, inner, axes):
    n = len(outer) + len(inner)
    outer_axes = filter(lambda i: i not in axes, set(range(n)))

    res = [0] * n

    for axe, v in zip(axes, inner):
        res[axe] = v

    for axe, v in zip(outer_axes, outer):
        res[axe] = v

    return tuple(res)


def ndarray_from_list(blocklist, grid):
    # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
    if isinstance(blocklist[0], np.ndarray):
        bs = np.empty(grid, dtype=object, order="F")
        for i, block in enumerate(blocklist):
            bs.flat[i] = block
        return bs.reshape(grid)
    else:
        return np.array(blocklist, order="F", dtype=type(blocklist[0])).reshape(grid)


def slice_blocks(eq, arrays, sliced_inds):
    import rosnet

    tmp_arrays = list(arrays)
    signatures = eq.split("->")[0].split(",")

    for i, signature in enumerate(signatures):
        if any(label in sliced_inds for label in signature):
            shape = tmp_arrays[i].shape
            blockshape = tuple(
                map(
                    lambda x: 1 if x[1] else shape[x[0]],
                    enumerate(label in sliced_inds for label in signature),
                )
            )
            tmp_arrays[i] = rosnet.array(tmp_arrays, blockshape=blockshape)

    return tmp_arrays
