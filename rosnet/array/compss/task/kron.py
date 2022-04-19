import numpy as np
from rosnet.core import log
from rosnet.core.abc import Array
from rosnet.tuning.task import autotune


@autotune(returns=1)
@log.trace
def kron(a: Array, b: Array) -> Array:
    return np.kron(a, b)
