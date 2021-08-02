from pycompss.api.task import task
from pycompss.api.parameter import IN, INOUT
import numpy as np


@task(block=IN)
def getitem(block: np.ndarray, idx):
    return block[idx]


@task(block=INOUT)
def setitem(block: np.ndarray, idx, value):
    block[idx] = value


@task(block=INOUT, returns=np.ndarray)
def reshape(block: np.ndarray, shape):
    # WARNING unsafe
    block.shape = shape


@task(block=IN, returns=np.ndarray)
def copy(block: np.ndarray):
    return np.copy(block)
