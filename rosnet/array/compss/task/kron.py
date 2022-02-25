import numpy as np
from rosnet.tuning.task import autotune


@autotune(returns=np.ndarray)
def kron(a: np.ndarray, b: np.ndarray):
    return np.kron(a, b)
