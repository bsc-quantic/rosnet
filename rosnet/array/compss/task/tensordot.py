from typing import Sequence, Union
import os
import numpy as np
from pycompss.api.parameter import IN, COLLECTION_IN, COMMUTATIVE, Type, Depth
from rosnet.tuning.task import autotune
from rosnet.array.maybe import MaybeArray


def _fix_blas_threads():
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
    os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]


@autotune(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def sequential(a: Sequence[Union[np.ndarray, "DataClayBlock"]], b: Sequence[Union[np.ndarray, "DataClayBlock"]], axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@autotune(ba=IN, bb=IN, returns=np.ndarray)
def tensordot(ba: Union[np.ndarray, "DataClayBlock"], bb: Union[np.ndarray, "DataClayBlock"], axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


# @tensordot.register(processors=[{"processorType": "GPU", "computingUnits": "1"}])
# def tensordot_gpu(ba: np.ndarray, bb: np.ndarray, axes):
#     _fix_blas_threads()
#     return do("tensordot", ba, bb, axes, like="cupy")


@autotune(res=COMMUTATIVE)
def commutative(res: Union[MaybeArray, np.ndarray], a: Union[np.ndarray, "DataClayBlock"], b: Union[np.ndarray, "DataClayBlock"], axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)
