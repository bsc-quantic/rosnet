from pycompss.api.task import task
from pycompss.api.parameter import IN, INOUT
import numpy as np


@autotune(block=IN)
def getitem(block: np.ndarray, idx):
    return block[idx]


@autotune(block=INOUT)
def setitem(block: np.ndarray, idx, value):
    block[idx] = value


@autotune(block=INOUT, returns=np.ndarray)
def reshape(block: np.ndarray, shape):
    # WARNING unsafe
    block.shape = shape


@autotune(block=IN, returns=np.ndarray)
def copy(block: np.ndarray):
    return np.copy(block)
