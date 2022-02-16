from typing import List
import os
import numpy as np
from pycompss.api.parameter import IN, COLLECTION_IN, COMMUTATIVE, Type, Depth
from rosnet.tuning.task import autotune
from rosnet.array.maybe import MaybeArray


def _fix_blas_threads():
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
    os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]


@autotune(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def sequential(a: List[np.ndarray], b: List[np.ndarray], axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@autotune(ba=IN, bb=IN, returns=np.ndarray)
def tensordot(ba: np.ndarray, bb: np.ndarray, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@autotune(res=COMMUTATIVE)
def commutative(res: MaybeArray, a: np.ndarray, b: np.ndarray, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)
