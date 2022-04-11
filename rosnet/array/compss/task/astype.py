import numpy as np
from pycompss.api.parameter import IN, INOUT
from rosnet.core import log
from rosnet.core.interface import Array
from rosnet.tuning.task import autotune


@autotune(a=IN, returns=1)
@log.trace
def astype_copy(a: Array, dtype: np.dtype, order="K", casting="unsafe", subok=True) -> Array:
    return a.astype(dtype, order=order, casting=casting, subok=subok, copy=True)
