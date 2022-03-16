from typing import Tuple
import numpy as np
from pycompss.api.parameter import IN
from rosnet.core.interface import Array
from rosnet.tuning.task import autotune


@autotune(a=IN, returns=2)
def qr_complete(a: Array) -> Tuple[Array, Array]:
    return np.linalg.qr(a, mode="complete")


@autotune(a=IN, returns=2)
def qr_reduced(a: Array) -> Tuple[Array, Array]:
    return np.linalg.qr(a, mode="reduced")


@autotune(a=IN, returns=1)
def qr_r(a: Array) -> Array:
    return np.linalg.qr(a, mode="r")


@autotune(a=IN, returns=2)
def qr_raw(a: Array) -> Tuple[Array, Array]:
    return np.linalg.qr(a, mode="raw")
