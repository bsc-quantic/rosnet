from contextlib import contextmanager
from typing import Tuple, Callable
from math import ceil
import numpy as np
from .util import core_count
from rosnet import helper
from rosnet.utils import result_shape, prod


class Tuner:
    def __init__(self, *args, **kwargs):
        self.max_cpu = kwargs.get("max_cpu") or core_count()
        self.threshold_flops = kwargs.get("threshold_flops") or 0
        self.commutative_threshold = kwargs.get("threshold_k") or 2

    def tensordot(self, a, b, axes) -> Tuple[Callable, int]:
        nblock = prod(map(lambda i: a.grid[i], axes[0]))
        typ = np.result_type(a.dtype, b.dtype)
        blockshape = result_shape(a.blockshape, b.blockshape, axes)
        blocknbytes = prod(blockshape) * typ.itemsize

        # choose implementation by means of number of blocks
        impl = helper.tensordot_sequential

        if nblock == 1:
            impl = helper.tensordot_tensordot

        if nblock > self.commutative_threshold:
            impl = helper.tensordot_commutative

        # choose parallelism conservatively based on calculated memory usage
        mem_usage = 0
        if impl == helper.tensordot_sequential:
            mem_usage = blocknbytes + nblock * (a.blocknbytes + b.blocknbytes)
        else:
            mem_usage = a.blocknbytes + b.blocknbytes + blocknbytes

        memory = 96 * 1024 ** 3
        par = min(int(ceil(core_count() * mem_usage / memory)), self.max_cpu)

        return (impl, par)


tuner = Tuner()


@contextmanager
def configure(*args, **kwargs):
    global tuner
    tmp = tuner

    tuner = Tuner(*args, **kwargs)

    yield

    tuner = tmp
