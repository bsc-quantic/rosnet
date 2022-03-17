import numpy as np
from pycompss.api.parameter import IN, INOUT
from rosnet.tuning.task import autotune
from rosnet.core.interface import Array


@autotune(a=IN, returns=1)
def cumsum(a: Array, axis=None, dtype=None) -> Array:
    return np.cumsum(a, axis=axis, dtype=dtype)


@autotune(a=IN, out=INOUT, returns=0)
def cumsum_out(out: Array, a: Array, axis=None, dtype=None):
    np.cumsum(a, axis=axis, dtype=dtype, out=out)
