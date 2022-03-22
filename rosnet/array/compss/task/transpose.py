import numpy as np
from pycompss.api.parameter import IN, INOUT
from rosnet.tuning.task import autotune
from rosnet.core.interface import Array


@autotune(block=IN, returns=Array)
def transpose(block: Array, axes) -> Array:
    return np.transpose(block, axes)


@autotune(block=INOUT)
def transpose_inplace(block: Array, axes):
    # WARNING unsafe
    # TODO before was np.asfortranarray
    tmp = np.ascontiguousarray(np.transpose(block, axes))
    # TODO PROBLEMA DE DOBLE SERIALIZACION
    block.shape = tmp.shape
    block.data = tmp.data


# @transpose.register(app_software="hptt")
# def __transpose_hptt(block: np.ndarray, axes):
#     hptt.tensorTransposeAndUpdate(block, axes)
