import functools
from rosnet.helper.typing import Future


class TunableTask:
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

    def __getitem__(self, **kwargs):
        return self.generate_variant(**kwargs)

    def __call__(self, *args, **kwargs):
        if self.fn is None:
            self.fn = args[0]
            return self

        # automatic unpacking of wrapper objects
        args = [arg.ref if isinstance(arg, Future) else arg for arg in args]

        return self.generate_variant()(*args, **kwargs)


tunable = TunableTask
