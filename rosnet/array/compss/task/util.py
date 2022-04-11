from pycompss.api.parameter import IN, INOUT
import numpy as np
from rosnet.tuning.task import autotune
from rosnet.core import log


@autotune(block=IN, returns=1)
@log.trace
def getitem(block: np.ndarray, idx):
    return block[idx]


@autotune(block=INOUT)
@log.trace
def setitem(block: np.ndarray, idx, value):
    block[idx] = value


@autotune(block=IN, returns=1)
@log.trace
def reshape(block: np.ndarray, shape, order) -> np.ndarray:
    return np.reshape(block, shape, order=order)


@autotune(block=INOUT, returns=0)
@log.trace
def reshape_inplace(block: np.ndarray, shape):
    block.shape = shape


@autotune(block=IN, returns=1)
@log.trace
def copy(block: np.ndarray) -> np.ndarray:
    return np.copy(block)
