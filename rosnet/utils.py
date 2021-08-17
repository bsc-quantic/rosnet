from typing import List
import functools
import itertools
import operator as op
import numpy as np
from pycompss.api.task import task

def isunique(l: List) -> bool:
    """ Checks whether all the elements in `l` are unique in `l` """
    return len(set(l)) == len(l)


def space(s: list):
    """ Generates an iterator through the Cartesian space of dimensionality `s` """
    return itertools.product(*[range(i) for i in s])


def ispower2(n: int):
    return (n & (n-1)) and n != 0


try:
    from math import prod
except ImportError:
    def prod(x):
        return functools.reduce(op.mul, x, 1)


def ndarray_from_list(blocklist, grid):
    # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
    if isinstance(blocklist[0], np.ndarray):
        bs = np.empty(grid, dtype=object, order='F')
        for i, block in enumerate(blocklist):
            bs.flat[i] = block
        return bs.reshape(grid)
    else:
        return np.array(blocklist, order='F', dtype=type(blocklist[0])).reshape(grid)


def todo(func):
    @functools.wraps(func)
    def todo_msg(*args, **kwargs):
        raise NotImplementedError(f'{func.__name__} is not implemented yet')
    return todo_msg


@todo
def generic_task(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return task(func, None, **kwargs)(*args)
    return wrapper(func)


# from https://github.com/dask/dask/blob/95fb60a31a87c6b94b01ed75ab6533fa04d51f19/dask/utils.py
def inherit_doc(parent):
    def wrapper(method):
        extra = getattr(method, '__doc__', None) or ''
        # parent = getattr(lib, method.__name__)
        method.__doc__ = parent.__doc__ or ''
        method.__doc__ += '\n\n' + extra.rstrip('\n')
        return method
    return wrapper


# from https://stackoverflow.com/a/18078819
class hybridmethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        context = obj if obj is not None else cls

        @functools.wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.func(context, *args, **kwargs)

        wrapper.__func__ = self.func
        wrapper.__self__ = context

        return wrapper

def isarray(o: object):
    return isinstance(o, np.ndarray) or issubclass(o, np.ndarray) or hasattr(o, '__array__')


def result_nblock(a, b, axes):
    "Returns the number of blocks of the resulting array after `tensordot`."
    return prod(
        prod(itertools.compress(grid, [x not in ax for x in range(len(grid))]))
        for ax, grid in zip(axes, (a.grid, b.grid))
    )


def result_shape(a, b, axes):
    "Returns the blockshape of the resulting array after `tensordot`."
    outer_axes = [tuple(set(range(len(bs))) - set(ax)) for ax, bs in zip(axes, (a, b))]
    return functools.reduce(
        op.add,
        (tuple(i[ax] for ax in outer_ax) for outer_ax, i in zip(outer_axes, (a, b))),
    )