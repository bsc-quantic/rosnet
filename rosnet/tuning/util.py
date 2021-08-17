import os
import multiprocessing
from pycompss.util.context import in_master, in_worker
from pycompss.api.api import compss_get_number_of_resources
from rosnet.array import result_shape
from rosnet.utils import prod

node_count = compss_get_number_of_resources


def core_count():
    "Return the number of compute units."
    if in_master():
        return os.environ["DEFAULT_CPUS_PER_NODE"]
    else:
        return multiprocessing.cpu_count()


def flops_tensordot(a, b, axes):
    outer_blockshape = list(result_shape(a, b, axes))
    inner_blockshape = [a[i] for i in axes[0]]
    return prod(outer_blockshape + inner_blockshape)
