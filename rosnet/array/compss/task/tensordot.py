from typing import Sequence, Union
import os
import numpy as np
from pycompss.api.parameter import IN, COLLECTION_IN, COMMUTATIVE, Type, Depth
from rosnet.core.interface import Array
from rosnet.tuning.task import autotune
from rosnet.array.maybe import MaybeArray


def _fix_blas_threads():
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
    os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]


@autotune(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def sequential(a: Sequence[Array], b: Sequence[Array], axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@autotune(ba=IN, bb=IN, returns=np.ndarray)
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


@autotune(res=COMMUTATIVE)
def commutative(res: Array, a: Array, b: Array, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)
