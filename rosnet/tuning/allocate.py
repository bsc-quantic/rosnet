from contextlib import contextmanager

# NOTE not thread-safe
NCORES = 1
MEMORY = None


@contextmanager
def allocate(**kwargs):
    # pylint: disable=global-statement
    global NCORES
    global MEMORY

    tmp_ncores = NCORES
    tmp_memory = MEMORY

    NCORES = kwargs.get("ncores") or tmp_ncores
    MEMORY = kwargs.get("memory") or tmp_memory

    __fix_dyn_par_tensordot()

    yield

    NCORES = tmp_ncores
    MEMORY = tmp_memory

    __fix_dyn_par_tensordot()


# temporary fix for dynamic parallelism
def __fix_dyn_par_tensordot():
    from rosnet.task import tensordot

    if NCORES == 1:
        tensordot.sequential = tensordot.sequential_1
        tensordot.tensordot = tensordot.tensordot_1
        tensordot.commutative = tensordot.commutative_1
    elif NCORES == 2:
        tensordot.sequential = tensordot.sequential_2
        tensordot.tensordot = tensordot.tensordot_2
        tensordot.commutative = tensordot.commutative_2
    elif 2 < NCORES <= 4:
        tensordot.sequential = tensordot.sequential_4
        tensordot.tensordot = tensordot.tensordot_4
        tensordot.commutative = tensordot.commutative_4
    elif 4 < NCORES <= 8:
        tensordot.sequential = tensordot.sequential_8
        tensordot.tensordot = tensordot.tensordot_8
        tensordot.commutative = tensordot.commutative_8
    elif 8 < NCORES <= 12:
        tensordot.sequential = tensordot.sequential_12
        tensordot.tensordot = tensordot.tensordot_12
        tensordot.commutative = tensordot.commutative_12
    elif 12 < NCORES <= 24:
        tensordot.sequential = tensordot.sequential_24
        tensordot.tensordot = tensordot.tensordot_24
        tensordot.commutative = tensordot.commutative_24
