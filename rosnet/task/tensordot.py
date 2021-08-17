import numpy as np
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.parameter import IN, COLLECTION_IN, COMMUTATIVE, Type, Depth


@constraint(computing_units="$NCORES", memory="$MEMORY")
@task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def sequential(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="$NCORES", memory_size="$MEMORY")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot(ba, bb, axes):
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="$NCORES", memory_size="$MEMORY")
@task(res=COMMUTATIVE)
def commutative(res, a, b, axes):
    res += np.tensordot(a, b, axes)


# TODO use tensordot(a, b, axes=0)
@constraint(computing_units="$NCORES", memory_size="$MEMORY")
@task(returns=np.ndarray)
def kron(a: np.ndarray, b: np.ndarray):
    return np.kron(a, b)
