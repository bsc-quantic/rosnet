import numpy as np
from rosnet.tuning.task import autotune
from rosnet.core.interface import Array


@autotune(returns=np.ndarray)
def kron(a: Array, b: Array) -> Array:
    return np.kron(a, b)
