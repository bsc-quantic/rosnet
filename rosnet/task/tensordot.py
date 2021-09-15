import numpy as np
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.parameter import IN, COLLECTION_IN, COMMUTATIVE, Type, Depth


# @constraint(computing_units="$NCORES", memory_size="$MEMORY")
# @task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
# def sequential(a, b, axes):
#     return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="1")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_1(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="2")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_2(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="4")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_4(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="8")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_8(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="12")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_12(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@constraint(computing_units="24")
@task(
    a={Type: COLLECTION_IN, Depth: 1},
    b={Type: COLLECTION_IN, Depth: 1},
    returns=np.ndarray,
)
def sequential_24(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


sequential = sequential_1


# @constraint(computing_units="$NCORES", memory_size="$MEMORY")
# @task(ba=IN, bb=IN, returns=np.ndarray)
# def tensordot(ba, bb, axes):
#     return np.tensordot(ba, bb, axes)


@constraint(computing_units="1")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_1(ba, bb, axes):
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="2")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_2(ba, bb, axes):
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="4")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_4(ba, bb, axes):
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="8")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_8(ba, bb, axes):
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="12")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_12(ba, bb, axes):
    return np.tensordot(ba, bb, axes)


@constraint(computing_units="24")
@task(ba=IN, bb=IN, returns=np.ndarray)
def tensordot_24(ba, bb, axes):
    return np.tensordot(ba, bb, axes)


tensordot = tensordot_1


# @constraint(computing_units="$NCORES", memory_size="$MEMORY")
# @task(res=COMMUTATIVE)
# def commutative(res, a, b, axes):
#     res += np.tensordot(a, b, axes)


@constraint(computing_units="1")
@task(res=COMMUTATIVE)
def commutative_1(res, a, b, axes):
    res += np.tensordot(a, b, axes)


@constraint(computing_units="2")
@task(res=COMMUTATIVE)
def commutative_2(res, a, b, axes):
    res += np.tensordot(a, b, axes)


@constraint(computing_units="4")
@task(res=COMMUTATIVE)
def commutative_4(res, a, b, axes):
    res += np.tensordot(a, b, axes)


@constraint(computing_units="8")
@task(res=COMMUTATIVE)
def commutative_8(res, a, b, axes):
    res += np.tensordot(a, b, axes)


@constraint(computing_units="12")
@task(res=COMMUTATIVE)
def commutative_12(res, a, b, axes):
    res += np.tensordot(a, b, axes)


@constraint(computing_units="24")
@task(res=COMMUTATIVE)
def commutative_24(res, a, b, axes):
    res += np.tensordot(a, b, axes)


commutative = commutative_1

# TODO use tensordot(a, b, axes=0)
@constraint(computing_units="$NCORES", memory_size="$MEMORY")
@task(returns=np.ndarray)
def kron(a: np.ndarray, b: np.ndarray):
    return np.kron(a, b)
