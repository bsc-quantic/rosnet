from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.implement import implement
from pycompss.api.parameter import IN
import numpy as np


@task(block=IN, returns=np.ndarray)
def transpose(block: np.ndarray, axes):
    # WARNING unsafe
    # TODO before was np.asfortranarray
    tmp = np.ascontiguousarray(np.transpose(block, axes))
    block.shape = tmp.shape
    block.data = tmp.data


@implement(source_class="rosnet.task", method="transpose")
@constraint(app_software="hptt")
@task(block=IN, returns=np.ndarray)
def __transpose_hptt(block: np.ndarray, axes):
    hptt.tensorTransposeAndUpdate(block, axes)
