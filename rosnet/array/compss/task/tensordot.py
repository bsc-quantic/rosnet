import os
from typing import Sequence, Union

import numpy as np
from pycompss.api.parameter import COLLECTION_IN, COMMUTATIVE, IN, Depth, Type
from rosnet.array.maybe import MaybeArray
from rosnet.core import log
from rosnet.core.abc import Array
from rosnet.tuning.task import autotune


def _fix_blas_threads():
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")


@autotune(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=1)
@log.trace
def sequential(a: Sequence[Array], b: Sequence[Array], axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@autotune(ba=IN, bb=IN, returns=1)
@log.trace
def tensordot(ba: Array, bb: Array, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


# @tensordot.register(processors=[{"processorType": "GPU"}])
# def tensordot_gpu():
#     _fix_blas_threads()
#     _tensordot_gpu()

# @multimethod
# def _tensordot_gpu(ba: np.ndarray, bb: np.ndarray, axes):
#     return do("tensordot", ba, bb, axes, like="cupy")

# @_tensordot_gpu.register
# def _tensordot_gpu(ba: DataClayBlock, bb: DataClayBlock, axes):
#     # TODO think how to execute on gpu


@autotune(res=COMMUTATIVE, returns=0)
@log.trace
def commutative(res: Array, a: Array, b: Array, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)
