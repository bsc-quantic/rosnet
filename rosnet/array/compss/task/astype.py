import numpy as np
from pycompss.api.parameter import IN, INOUT
from rosnet.array.compss import COMPSsArray
from rosnet.tuning.task import autotune
from rosnet.core.interface import Array


@autotune(a=IN)
def astype_copy(a: Array, dtype: np.dtype, order="K", casting="unsafe", subok=True) -> Array:
    return a.astype(dtype, order=order, casting=casting, subok=subok, copy=True)
