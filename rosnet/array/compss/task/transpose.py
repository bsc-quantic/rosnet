import numpy as np
from pycompss.api.parameter import IN, INOUT
from rosnet.tuning.task import autotune
from rosnet.core import log
from rosnet.core.interface import Array


@autotune(block=IN, returns=Array)
@log.trace
def transpose(block: Array, axes) -> Array:
    return np.transpose(block, axes)


@autotune(block=INOUT)
@log.trace
def transpose_inplace(block: Array, axes):
    block_aux = np.transpose(block, axes)

    block.shape = block_aux.shape
    np.copyto(block, block_aux)


# @transpose.register(app_software="hptt")
# def __transpose_hptt(block: np.ndarray, axes):
#     hptt.tensorTransposeAndUpdate(block, axes)
