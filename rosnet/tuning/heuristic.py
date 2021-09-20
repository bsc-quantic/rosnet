from abc import abstractmethod
from contextlib import contextmanager
from typing import Tuple, Callable
import numpy as np
from .util import core_count, node_count, flops_tensordot
from rosnet import helper
from rosnet.utils import result_nblock, result_shape, prod

DEFAULT_THRESHOLD_FLOPS = 2 ** 20


class Heuristic:
    def __init__(self, *args, **kwargs):
        self.max_cpu = kwargs.get("max_cpu") or core_count()
        self.threshold_flops = kwargs.get("threshold_flops") or DEFAULT_THRESHOLD_FLOPS
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
        par = min(memory // mem_usage, self.max_cpu)

        return (impl, par)

    @abstractmethod
    def estimate(self, flops, flops_block, nblock) -> int:
        pass


class Default(Heuristic):
    def estimate(self, *args) -> int:
        return 1


class Eager(Heuristic):
    def estimate(self, flops, flops_block, nblock) -> int:
        "Returns the number of cores to be used by the task."

        available_parallelism = node_count() * core_count()

        if nblock > available_parallelism:
            return 1

        ncores = min(available_parallelism // nblock, self.max_cpu)
        return ncores


class Fixed(Heuristic):
    def __init__(self, ncores, **kwargs):
        kwargs["max_cpu"] = ncores
        super().__init__(**kwargs)
        self.ncores = ncores

    def estimate(self, *args) -> int:
        return self.ncores


algorithm = Default()


@contextmanager
def heuristic(cls, *args, **kwargs):
    if not issubclass(cls, Heuristic):
        raise TypeError("'cls' must implement Heuristic")

    global algorithm
    tmp = algorithm
    algorithm = cls(*args, **kwargs)

    yield

    algorithm = tmp
