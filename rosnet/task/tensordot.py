import os
import numpy as np
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.parameter import IN, COLLECTION_IN, COMMUTATIVE, Type, Depth


class ArrayWrapper(object):
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
            self.__init = True

    def __getstate__(self):
        return {"init": self.__init, "array": self.__array}

    def __setstate__(self, d):
        self.__init = d["init"]
        self.__array = d["array"]


def _fix_blas_threads():
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
    os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]


# @constraint(computing_units="$NCORES", memory_size="$MEMORY")
# @task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
# def sequential(a, b, axes):
#     return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


def dispatch_sequential(a: List, b: List, axes):
    # pylint: disable=protected-access
    import rosnet

    blockshape = result_shape(a[0].shape, b[0].shape, axes)
    dtype = np.result_type(
        *(i.dtype if hasattr(i, "dtype") else i for i in chain(a, b))
    )

    # get ref if a,b are COMPSsArrays
    aref = [i._ref if hasattr(i, "_ref") else i for i in a]
    bref = [i._ref if hasattr(i, "_ref") else i for i in b]

    ref = rosnet.task.tensordot.sequential(aref, bref, axes)
    return rosnet.COMPSsArray(ref, shape=blockshape, dtype=dtype)


@constraint(computing_units="1")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_1(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="2")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_2(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="4")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_4(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="8")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_8(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="12")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_12(a, b, axes):
    _fix_blas_threads()
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="24")
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


def dispatch_tensordot(a: List, b: List, axes):
    # pylint: disable=protected-access
    import rosnet

    if len(a) > 1 or len(b) > 1:
        raise ValueError()

    blockshape = result_shape(a[0].shape, b[0].shape, axes)
    dtype = np.result_type(
        *(i.dtype if hasattr(i, "dtype") else i for i in chain(a, b))
    )

    # get ref if a,b are COMPSsArrays
    aref = [i._ref if hasattr(i, "_ref") else i for i in a][0]
    bref = [i._ref if hasattr(i, "_ref") else i for i in b][0]

    ref = rosnet.task.tensordot.tensordot(aref, bref, axes)
    return rosnet.COMPSsArray(ref, shape=blockshape, dtype=dtype)


@constraint(computing_units="1")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_1(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="2")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_2(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="4")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_4(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="8")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_8(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="12")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_12(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="24")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_24(ba, bb, axes):
    _fix_blas_threads()
    return np.tensordot(ba, bb, axes)


tensordot = tensordot_1


# @constraint(computing_units="$NCORES", memory_size="$MEMORY")
# @task(res=COMMUTATIVE)
# def commutative(res, a, b, axes):
#     res += np.tensordot(a, b, axes)


def dispatch_commutative(a: List, b: List, axes):
    # pylint: disable=protected-access
    import rosnet

    blockshape = result_shape(a[0].shape, b[0].shape, axes)
    dtype = np.result_type(
        *(i.dtype if hasattr(i, "dtype") else i for i in chain(a, b))
    )

    buffer = rosnet.task.tensordot.ArrayWrapper()

    for ia, ib in zip(a, b):
        # get ref if a,b are COMPSsArrays
        ia = ia._ref if hasattr(ia, "_ref") else ia
        ib = ib._ref if hasattr(ib, "_ref") else ib

        rosnet.task.tensordot.commutative(buffer, ia, ib, axes)

    return rosnet.COMPSsArray(buffer, shape=blockshape, dtype=dtype)


@constraint(computing_units="1")
@task(res=COMMUTATIVE)
def commutative_1(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="2")
@task(res=COMMUTATIVE)
def commutative_2(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="4")
@task(res=COMMUTATIVE)
def commutative_4(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="8")
@task(res=COMMUTATIVE)
def commutative_8(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="12")
@task(res=COMMUTATIVE)
def commutative_12(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


@constraint(computing_units="24")
@task(res=COMMUTATIVE)
def commutative_24(res: ArrayWrapper, a, b, axes):
    _fix_blas_threads()
    res += np.tensordot(a, b, axes)


commutative = commutative_1
