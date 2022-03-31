from pycompss.api.parameter import IN, INOUT
import numpy as np
from rosnet.tuning.task import autotune


@autotune(block=IN)
def getitem(block: np.ndarray, idx):
    return block[idx]


@autotune(block=INOUT)
def setitem(block: np.ndarray, idx, value):
    block[idx] = value


@autotune(block=IN, returns=np.ndarray)
def reshape(block: np.ndarray, shape, order):
    return np.reshape(block, shape, order=order)


@autotune(block=INOUT)
def reshape_inplace(block: np.ndarray, shape):
    block.shape = shape


@autotune(block=IN, returns=np.ndarray)
def copy(block: np.ndarray):
    return np.copy(block)
