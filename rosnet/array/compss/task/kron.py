import numpy as np
from rosnet.core import log
from rosnet.core.interface import Array
from rosnet.tuning.task import autotune


@autotune(returns=np.ndarray)
@log.trace
def kron(a: Array, b: Array) -> Array:
    return np.kron(a, b)
