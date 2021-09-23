import os
import numpy as np
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.parameter import IN, COLLECTION_IN, COMMUTATIVE, Type, Depth


def ArrayWrapper(object):
    def __init__(self):
        self.__init = False
        self.__array = None

    @property
    def initialized(self):
        return self.__init

    def __array__(self):
        return self.__array

    def __iadd__(self, rhs):
        if self.initialized:
            self.__array += rhs
        else:
            self.__array = rhs


def _fix_blas_threads():
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
    os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]


# @constraint(computing_units="$NCORES", memory_size="$MEMORY")
# @task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
# def sequential(a, b, axes):
#     return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="1", memory_size="2")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_1(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="2", memory_size="4")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_2(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="4", memory_size="8")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_4(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="8", memory_size="16")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_8(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="12", memory_size="24")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_12(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="24", memory_size="45")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_24(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


sequential = sequential_1


# @constraint(computing_units="$NCORES", memory_size="$MEMORY")
# @task(ba=IN, bb=IN, returns=np.ndarray)
# def tensordot(ba, bb, axes):
#     return np.tensordot(ba, bb, axes)


@constraint(computing_units="1", memory_size="2")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_1(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="2", memory_size="4")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_2(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="4", memory_size="8")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_4(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="8", memory_size="16")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_8(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="12", memory_size="24")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_12(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="24", memory_size="45")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_24(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


tensordot = tensordot_1


# @constraint(computing_units="$NCORES", memory_size="$MEMORY")
# @task(res=COMMUTATIVE)
# def commutative(res, a, b, axes):
#     res += np.tensordot(a, b, axes)


@constraint(computing_units="1", memory_size="2")
@task(res=COMMUTATIVE)
def commutative_1(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="2", memory_size="4")
@task(res=COMMUTATIVE)
def commutative_2(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="4", memory_size="8")
@task(res=COMMUTATIVE)
def commutative_4(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="8", memory_size="16")
@task(res=COMMUTATIVE)
def commutative_8(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="12", memory_size="24")
@task(res=COMMUTATIVE)
def commutative_12(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="24", memory_size="45")
@task(res=COMMUTATIVE)
def commutative_24(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


commutative = commutative_1

# TODO use tensordot(a, b, axes=0)
@constraint(computing_units="$NCORES", memory_size="$MEMORY")
@task(returns=np.ndarray)
def kron(a: np.ndarray, b: np.ndarray):
    return np.kron(a, b)
