import numpy as np
from pycompss.api.api import compss_get_number_of_resources
import multiprocessing


node_count = compss_get_number_of_resources
core_count = multiprocessing.cpu_count


class Default:
    def __call__(self, *args, **kwargs) -> int:
        return 1


class Eager:
    def __init__(self, **kwargs):
        self.threshold_flops = kwargs.get("threshold_flops") or 1
        self.max_core_per_task = kwargs.get("max_core_per_task") or core_count()

    def __call__(self, parallel_tasks, block_flops, *args, **kwargs) -> int:
        """Returns the number of cores to be used by the task.

        `parallel_tasks`: Number of independent task to be launched.
        """
        if block_flops < self.threshold_flops:
            print("block_flops < self.threshold_flops")
            return 1

        available_parallelism = node_count() * core_count()
        if parallel_tasks > available_parallelism:
            print("parallel_tasks > available_parallelism")
            return 1

        ncores = min(available_parallelism // parallel_tasks, core_count())
        print("[DEBUG] ncores=%r" % ncores)
        return ncores
