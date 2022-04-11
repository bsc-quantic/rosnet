from typing import Union

import numpy as np
from pycompss.api.parameter import IN
from rosnet.core import log
from rosnet.core.interface import Array
from rosnet.tuning.task import autotune


@autotune(a=IN, returns=1)
@log.trace
def count_nonzero(a: Array, axis, keepdims) -> Union[int, np.ndarray]:
    return np.count_nonzero(a, axis=axis, keepdims=keepdims)
