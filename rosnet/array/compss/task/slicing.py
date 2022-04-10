from typing import Sequence
import numpy as np
from pycompss.api.parameter import IN, COLLECTION_IN, COLLECTION_OUT, Type, Depth
from rosnet.core import log
from rosnet.core.interface import Array
from rosnet.tuning.task import autotune


@autotune(block=IN, returns={Type: COLLECTION_OUT, Depth: 1})
@log.trace
def split(block: Array, n: int, axis: int):
    return map(lambda x: x.copy(), np.split(block, n, axis))


@autotune(arrays={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
@log.trace
def stack(arrays: Sequence[Array], axis=0, out=None) -> np.ndarray:
    return np.stack(arrays, axis=axis, out=out)
