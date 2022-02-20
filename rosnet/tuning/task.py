from typing import Callable, Dict
import functools
from math import ceil
from rosnet.helper.typing import Future
from rosnet.helper.macros import todo

# TODO how to do @implements on this? maybe add an extra .dispatch/.register for multiple implementations? multi-gpu?
# TODO save new ""task"" in self? so we know which fn to generate in generate_variant for multi-gpu
# TODO how to skip this if local execution?
# TODO how to write numerical methods that do iterations -> support nested tasks -> how to wrap this methods for local and distributed execution?
class autotune:
    def __init__(self, **kwargs):
        self.task_info = kwargs
        self.__fn = None

    @property
    def fn(self):
        return self.__fn

    @functools.lru_cache
    def generate_variant(self, **kwargs):
        from pycompss.api.task import task
        from pycompss.api.constraint import constraint

        return constraint(**kwargs)(task(**self.task_info)(self.fn))

    @todo
    def register(self, *args, **kwargs):
        """Registers another implementation of the COMPSs task. e.g. GPU impl., implementation if some library is available"""

        def registrar(fn):
            from pycompss.api.task import task
            from pycompss.api.constraint import constraint
            from pycompss.api.implements import pycompss_implements

            pycompss_implements(source_class=self.fn.__module__, method=self.fn.__name__)(constraint(**kwargs)(task(**self.task_info)(fn)))
            return self

        return registrar

    def __getitem__(self, **kwargs):
        return self.generate_variant(**kwargs)

    def __call__(self, *args, **kwargs):
        if self.fn is None:
            self.fn = args[0]

            # generate primer task, need to register task before any call autotune.register
            _ = self.generate_variant()

            return self

        constraints = tune(self.fn, *args, **kwargs)

        # automatic unpacking of wrapper objects
        args = [arg.ref if isinstance(arg, Future) else arg for arg in args]

        return self.generate_variant(**constraints)(*args, **kwargs)


def tune(fn: Callable, *args, **kwargs) -> Dict[str, int]:
    from rosnet.tuning.compss import core_count

    # dispatch cost function
    try:
        mem_usage = do(fn.__name__, *args, **kwargs, like="rosnet.tuning.cost.mem")
    except:
        mem_usage = 1

    # TODO get memory space from COMPSs
    available_memory_per_node = 92 * 1024 ** 3

    # conservative parallelization: with uniform mem/core, min #cores that fulfill the task
    par = min(int(ceil(core_count() * mem_usage / available_memory_per_node)), 1)

    return {"computing_units": str(par)}
