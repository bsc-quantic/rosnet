from typing import List
from functools import reduce
from itertools import product


def isunique(l: List) -> bool:
    """ Checks whether all the elements in `l` are unique in `l` """
    return len(set(l)) == len(l)


def space(s: list):
    """ Generates an iterator through the Cartesian space of dimensionality `s` """
    return product(*[range(i) for i in s])


try:
    from math import prod
except ImportError:
    import operator

    def prod(x):
        return reduce(operator.mul, x, 1)

# def coord(s):
#     c = [s // stride[-1]]
#     mod = s % stride[-1]
#     for stride_j in reversed(stride[:-1]):
#         c.append(mod // stride_j)
#         mod = mod % stride_j
#     c.reverse()
#     return c
