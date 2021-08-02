import numpy as np
from pycompss.api.task import task


@task(returns=np.ndarray)
def kron(a: np.ndarray, b: np.ndarray):
    return np.kron(a, b)
