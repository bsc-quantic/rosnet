import numpy as np
from pycompss.api.parameter import IN, INOUT
from rosnet.core import log
from rosnet.core.abc import Array
from rosnet.tuning.task import autotune


@autotune(a=IN, returns=1)
@log.trace
def cumsum(a: Array, axis=None, dtype=None) -> Array:
    return np.cumsum(a, axis=axis, dtype=dtype)


@autotune(a=IN, out=INOUT, returns=0)
@log.trace
def cumsum_out(out: Array, a: Array, axis=None, dtype=None):
    np.cumsum(a, axis=axis, dtype=dtype, out=out)
