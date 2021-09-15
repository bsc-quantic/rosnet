from contextlib import contextmanager
from os import environ

# NOTE not thread-safe
# NCORES = 1
# MEMORY = None
environ["NCORES"] = "1"
environ["MEMORY"] = "0"


# @contextmanager
# def resources(**kwargs):
#     # pylint: disable=global-statement
#     # TODO COMPSs crashes with global variables
#     global NCORES
#     global MEMORY

#     tmp = {
#         "NCORES": NCORES,
#         "MEMORY": MEMORY,
#     }

#     NCORES = kwargs.get("ncores") or tmp["NCORES"]
#     MEMORY = kwargs.get("memory") or tmp["MEMORY"]

#     yield

#     NCORES = tmp["NCORES"]
#     MEMORY = tmp["MEMORY"]


@contextmanager
def resources(**kwargs):
    tmp = {
        "NCORES": environ["NCORES"],
        "MEMORY": environ["MEMORY"],
    }

    environ["NCORES"] = str(kwargs.get("ncores") or tmp["NCORES"])
    environ["MEMORY"] = str(kwargs.get("memory") or tmp["MEMORY"])

    __fix_dyn_par_tensordot()

    yield

    environ["NCORES"] = tmp["NCORES"]
    environ["MEMORY"] = tmp["MEMORY"]

    __fix_dyn_par_tensordot()


# temporary fix for dynamic parallelism
def __fix_dyn_par_tensordot():
    from rosnet.task import tensordot

    if int(environ["NCORES"]) == 1:
        tensordot.sequential = tensordot.sequential_1
        tensordot.tensordot = tensordot.tensordot_1
        tensordot.commutative = tensordot.commutative_1
    elif int(environ["NCORES"]) == 2:
        tensordot.sequential = tensordot.sequential_2
        tensordot.tensordot = tensordot.tensordot_2
        tensordot.commutative = tensordot.commutative_2
    elif 2 < int(environ["NCORES"]) <= 4:
        tensordot.sequential = tensordot.sequential_4
        tensordot.tensordot = tensordot.tensordot_4
        tensordot.commutative = tensordot.commutative_4
    elif 4 < int(environ["NCORES"]) <= 8:
        tensordot.sequential = tensordot.sequential_8
        tensordot.tensordot = tensordot.tensordot_8
        tensordot.commutative = tensordot.commutative_8
    elif 8 < int(environ["NCORES"]) <= 12:
        tensordot.sequential = tensordot.sequential_12
        tensordot.tensordot = tensordot.tensordot_12
        tensordot.commutative = tensordot.commutative_12
    elif 12 < int(environ["NCORES"]) <= 24:
        tensordot.sequential = tensordot.sequential_24
        tensordot.tensordot = tensordot.tensordot_24
        tensordot.commutative = tensordot.commutative_24
