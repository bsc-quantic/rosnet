import numpy as np
from rosnet.tuning.task import autotune
from pycompss.api.parameter import IN


@autotune(block=IN, returns=np.ndarray)
def transpose(block: np.ndarray, axes):
    # WARNING unsafe
    # TODO before was np.asfortranarray
    tmp = np.ascontiguousarray(np.transpose(block, axes))
    block.shape = tmp.shape
    block.data = tmp.data


@transpose.register(app_software="hptt")
def __transpose_hptt(block: np.ndarray, axes):
    hptt.tensorTransposeAndUpdate(block, axes)
