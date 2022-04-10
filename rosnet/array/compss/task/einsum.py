import numpy as np
from pycompss.api.parameter import IN, INOUT
from rosnet.core import log
from rosnet.core.interface import Array
from rosnet.tuning.task import autotune


@autotune(operands=IN)
@log.trace
def einsum(pattern: str, *operands, dtype=None, order="K", casting="safe", optimize=False) -> Array:
    return np.einsum(pattern, *operands, dtype=dtype, order=order, casting=casting, optimize=optimize)


# TODO what if operands[i] == out?
# TODO out = OUT dependency?
@autotune(operands=IN, out=INOUT)
@log.trace
def einsum_out(pattern: str, *operands, dtype=None, order="K", casting="safe", optimize=False):
    return np.einsum(pattern, *operands, dtype=dtype, order=order, casting=casting, optimize=optimize)
