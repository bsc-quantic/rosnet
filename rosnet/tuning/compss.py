from pycompss.api.api import compss_get_number_of_resources
from pycompss.util.context import in_master, in_worker

node_count = compss_get_number_of_resources


def _core_count():
    "Return the number of compute units."
    if in_master():
        import os

        return os.cpu_count()
    else:
        import multiprocessing

        return multiprocessing.cpu_count()
