from typing import List
from functools import reduce
from itertools import product
import numpy as np


def isunique(l: List) -> bool:
    """ Checks whether all the elements in `l` are unique in `l` """
    return len(set(l)) == len(l)


def space(s: list):
    """ Generates an iterator through the Cartesian space of dimensionality `s` """
    return product(*[range(i) for i in s])


def ispower2(n: int):
    return (n & (n-1)) and n != 0


try:
    from math import prod
except ImportError:
    import operator

    def prod(x):
        return reduce(operator.mul, x, 1)


def ndarray_from_list(blocklist, grid):
    # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
    if isinstance(blocklist[0], np.ndarray):
        bs = np.empty(grid, dtype=object, order='F')
        for i, block in enumerate(blocklist):
            bs.flat[i] = block
        return bs.reshape(grid)
    else:
        return np.array(blocklist, order='F').reshape(grid)
