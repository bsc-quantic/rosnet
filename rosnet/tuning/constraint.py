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

    yield

    environ["NCORES"] = tmp["NCORES"]
    environ["MEMORY"] = tmp["MEMORY"]
