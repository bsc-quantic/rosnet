from typing import Tuple

import numpy as np
from pycompss.api.parameter import IN
from rosnet.core import log
from rosnet.core.abc import Array
from rosnet.tuning.task import autotune


@autotune(a=IN, returns=2)
@log.trace
def qr_complete(a: Array) -> Tuple[Array, Array]:
    return np.linalg.qr(a, mode="complete")


@autotune(a=IN, returns=2)
@log.trace
def qr_reduced(a: Array) -> Tuple[Array, Array]:
    return np.linalg.qr(a, mode="reduced")


@autotune(a=IN, returns=1)
@log.trace
def qr_r(a: Array) -> Array:
    return np.linalg.qr(a, mode="r")


@autotune(a=IN, returns=2)
@log.trace
def qr_raw(a: Array) -> Tuple[Array, Array]:
    return np.linalg.qr(a, mode="raw")
