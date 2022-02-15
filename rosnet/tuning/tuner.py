from contextlib import contextmanager
from typing import Tuple, Callable
from math import ceil, prod
import numpy as np


class Tuner:
    def __init__(self, *args, **kwargs):
        self.max_cpu = kwargs.get("max_cpu", _core_count())
        self.threshold_flops = kwargs.get("threshold_flops", 0)
        self.commutative_threshold = kwargs.get("threshold_k", 2)

    def tensordot(self, a, b, axes) -> Tuple[Callable, int]:
        nblock = prod(map(lambda i: a.grid[i], axes[0]))
        typ = np.result_type(a.dtype, b.dtype)
        blockshape = result_shape(a.blockshape, b.blockshape, axes)
        blocknbytes = prod(blockshape) * typ.itemsize

        # choose implementation by means of number of blocks
        impl = dispatch_sequential

        if nblock == 1:
            impl = dispatch_tensordot

        if nblock > self.commutative_threshold:
            impl = dispatch_commutative

        # choose parallelism conservatively based on calculated memory usage
        mem_usage = 0
        if impl == dispatch_sequential:
            mem_usage = blocknbytes + nblock * (a.blocknbytes + b.blocknbytes)
        elif impl == dispatch_commutative:
            # NOTE without the x2 factor, it goes Out-Of-Memory
            # don't know why this happens
            mem_usage = 2 * (a.blocknbytes + b.blocknbytes + blocknbytes)
        else:
            mem_usage = a.blocknbytes + b.blocknbytes + blocknbytes

        memory = 92 * 1024 ** 3
        par = min(int(ceil(core_count() * mem_usage / memory)), self.max_cpu)

        return (impl, par)


__tuner = Tuner()


@contextmanager
def configure(*args, **kwargs):
    global __tuner
    tmp = __tuner

    __tuner = Tuner(*args, **kwargs)

    yield

    __tuner = tmp


def tensordot(a, b, axes):
    global __tuner
    return __tuner.tensordot(a, b, axes)
