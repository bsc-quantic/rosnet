from pycompss.api.parameter import IN, INOUT
import numpy as np
from rosnet.tuning.task import autotune
from rosnet.core import log


@autotune(block=IN)
@log.trace
def getitem(block: np.ndarray, idx):
    return block[idx]


@autotune(block=INOUT)
@log.trace
def setitem(block: np.ndarray, idx, value):
    block[idx] = value


@autotune(block=IN, returns=np.ndarray)
@log.trace
def reshape(block: np.ndarray, shape, order):
    return np.reshape(block, shape, order=order)


@autotune(block=INOUT)
@log.trace
def reshape_inplace(block: np.ndarray, shape):
    block.shape = shape


@autotune(block=IN, returns=np.ndarray)
@log.trace
def copy(block: np.ndarray):
    return np.copy(block)
