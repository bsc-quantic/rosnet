from concurrent.futures import Executor
import sys
from typing import TypeVar, Callable, Generic
import functools
from pycompss.runtime.management.classes import Future

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


class COMPSsExecutor(Executor):
    def submit(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Future[T]:  # type: ignore
        task = COMPSsExecutor.variant_task(fn, ...)
        return task(*args, **kwargs)

    @staticmethod
    @functools.lru_cache(maxsize=0)
    def variant_task(fn: Callable[P, T], resources) -> Callable[P, Future[T]]:
        pass
