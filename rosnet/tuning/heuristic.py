from abc import abstractmethod
from contextlib import contextmanager
from typing import Tuple, Callable
from .util import core_count, node_count, flops_tensordot
from rosnet import helper
from rosnet.utils import result_nblock

DEFAULT_THRESHOLD_FLOPS = 2 ** 20


class Heuristic:
    def __init__(self, *args, **kwargs):
        self.max_cpu = kwargs.get("max_cpu") or core_count()
        self.threshold_flops = kwargs.get("threshold_flops") or DEFAULT_THRESHOLD_FLOPS

    def tensordot(self, a, b, axes) -> Tuple[Callable, int]:
        flops = flops_tensordot(a.shape, b.shape, axes)
        flops_block = flops_tensordot(a.blockshape, b.blockshape, axes)
        nblock = result_nblock(a, b, axes)

        # choose implementation by grid and flops
        impl = None
        if a.nblock == 1 or b.nblock == 1:
            impl = helper.tensordot_sequential
        else:
            if flops_block < self.threshold_flops:
                impl = helper.tensordot_sequential
            else:
                impl = helper.tensordot_commutative

        # parallelize if threshold surpassed
        par = 1
        if flops_block > self.threshold_flops:
            par = min(self.estimate(flops, flops_block, nblock), self.max_cpu)

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
