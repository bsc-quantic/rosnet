from typing import List
from functools import reduce


def isunique(l: List) -> bool:
    """ Checks whether all the elements in `l` are unique in `l` """
    return len(set(l)) == len(l)


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
