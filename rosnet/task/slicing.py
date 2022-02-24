import numpy as np
from pycompss.api.parameter import IN, COLLECTION_IN, COLLECTION_OUT, Type, Depth
from rosnet.tuning.task import autotune


@autotune(block=IN, returns={Type: COLLECTION_OUT, Depth: 1})
def split(block: np.ndarray, n: int, axis: int):
    return map(lambda x: x.copy(), np.split(block, n, axis))


@autotune(blocks=COLLECTION_IN, returns=np.ndarray)
def merge(blocks, axis: int):
    return np.stack(blocks, axis)
