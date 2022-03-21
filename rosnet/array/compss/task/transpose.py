import numpy as np
from rosnet.tuning.task import autotune
from pycompss.api.parameter import IN, INOUT


@autotune(block=IN, returns=np.ndarray)
def transpose(block: np.ndarray, axes):
    return np.transpose(block, axes)


@autotune(block=INOUT)
def transpose_inplace(block: np.ndarray, axes):
    # WARNING unsafe
    # TODO before was np.asfortranarray
    tmp = np.ascontiguousarray(np.transpose(block, axes))
    # TODO PROBLEMA DE DOBLE SERIALIZACION
    block.shape = tmp.shape
    block.data = tmp.data


# @transpose.register(app_software="hptt")
# def __transpose_hptt(block: np.ndarray, axes):
#     hptt.tensorTransposeAndUpdate(block, axes)
