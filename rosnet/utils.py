from typing import List
from functools import reduce
from itertools import product


def isunique(l: List) -> bool:
    """ Checks whether all the elements in `l` are unique in `l` """
    return len(set(l)) == len(l)


def space(s: list):
    """ Generates an iterator through the Cartesian space of dimensionality `s` """
    return product(*[range(i) for i in s])


def ispower2(v: int):
    return (n & (n-1)) and n != 0


try:
    from math import prod
except ImportError:
    import operator

    def prod(x):
        return reduce(operator.mul, x, 1)
