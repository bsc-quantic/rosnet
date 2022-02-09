from contextlib import contextmanager
from typing import Tuple, Callable
from math import ceil, prod
import numpy as np
from pycompss.api.api import compss_get_number_of_resources
from pycompss.util.context import in_master, in_worker
from rosnet import helper
from rosnet.utils import result_shape


node_count = compss_get_number_of_resources


def _core_count():
    "Return the number of compute units."
    if in_master():
        import os

        return os.cpu_count()
    else:
        import multiprocessing

        return multiprocessing.cpu_count()


def _flops_tensordot(a, b, axes):
    outer_blockshape = list(result_shape(a, b, axes))
    inner_blockshape = [a[i] for i in axes[0]]
    return prod(outer_blockshape + inner_blockshape)


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
        impl = helper.tensordot_sequential

        if nblock == 1:
            impl = helper.tensordot_tensordot

        if nblock > self.commutative_threshold:
            impl = helper.tensordot_commutative

        # choose parallelism conservatively based on calculated memory usage
        mem_usage = 0
        if impl == helper.tensordot_sequential:
            mem_usage = blocknbytes + nblock * (a.blocknbytes + b.blocknbytes)
        elif impl == helper.tensordot_commutative:
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
