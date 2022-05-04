import functools
import inspect
import sys
from concurrent.futures import Executor
from concurrent.futures import Future as PyFuture
from typing import Callable, Generic, TypeVar

import numpy as np
from pycompss.api.api import compss_delete_object, compss_wait_on
from pycompss.api.constraint import constraint
from pycompss.api.parameter import INOUT, _Param
from pycompss.api.task import task
from pycompss.runtime.management.classes import Future as COMPSsFuture

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


class Future(PyFuture):
    def __init__(self, future):
        if not isinstance(future, COMPSsFuture):
            raise TypeError(f"future must of type COMPSsFuture but {type(future)}")

        self.future = future

    def cancel(self):
        compss_delete_object(self.future)

    def add_done_callback(self, fn) -> None:
        raise NotImplementedError("COMPSs does not support completion callbacks")


class COMPSsExecutor(Executor):
    def submit(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Future[T]:  # type: ignore
        # TODO infer output type
        resources = {}  # TODO call tuner
        task = COMPSsExecutor.variant_task(fn, args, kwargs, resources)  # type: ignore
        return task(*args, **kwargs)

    @staticmethod
    @functools.lru_cache(maxsize=0)
    def variant_task(fn: Callable[P, T], args, kwargs, resources) -> Callable[P, Future[T]]:  # type: ignore
        directions = task_directions(fn, *args, **kwargs)
        return constraint(**resources)(task(**directions)(fn))  # type: ignore


def full_qualname(obj) -> list[str]:
    # NOTE design decision: forget ufunc instance and inform that it is a ufunc, because for pattern matching we are more interested in that it is a ufunc that in the semantics (interface should be nevertheless the same because they are ufuncs)

    assert hasattr(obj, "__call__"), "'obj' has to be Callable"

    if inspect.isfunction(obj) or inspect.ismethod(obj):
        # functions
        # e.g. numpy.tensordot => ["numpy", "tensordot"]
        mod = obj.__module__.split(".")
        qualname = getattr(obj, "__qualname__", obj.__name__).split(".")

    elif inspect.isbuiltin(obj) and hasattr(obj, "__self__"):
        # ufunc methods
        # e.g. numpy.add.at => ["numpy", "ufunc", "at"]
        mod = obj.__self__.__class__.__module__.split(".")
        qualname = obj.__qualname__.split(".")

    else:
        # ufuncs or function objects
        # e.g. numpy.add => ["numpy", "ufunc"]
        mod = obj.__class__.__module__.split(".")
        qualname = getattr(obj.__class__, "__qualname__", obj.__class__.__name__).split(".")  # type: ignore

    return mod + qualname


# TODO do numpy's built-in methods (impl. in Cython) work?
def task_directions(fn: Callable, *args, **kwargs) -> dict[str, int | _Param]:
    module_path = full_qualname(fn)
    qualname = ".".join(module_path)
    match module_path:
        # builtins
        case ["builtins", "setitem"]:
            return {"returns": 0}

        case ["builtins", x] if x.startswith("i"):
            # NOTE in-place builtin operators return the value too
            raise NotImplementedError(f"'{fn}' acts in-place but COMPSs direction cannot be mark correctly as it is a built-in method.")

        case ["builtins", _]:
            return {"returns": 1}

        # numpy
        case ["numpy", "copyto"]:
            # NOTE just in case, not sure if useful
            return {"returns": 0, "dst": INOUT}

        case ["numpy", "ufunc", "at"]:
            raise NotImplementedError(f"'{fn.__self__}' acts in-place but COMPSs direction cannot be mark correctly as it is a built-in method.")

        case ["numpy", "ufunc", "reduceat"]:
            raise NotImplementedError(f"'{fn.__self__}' acts in-place but COMPSs direction cannot be mark correctly as it is a built-in method.")

        # inplace methods: cumsum, einsum, dot, linalg.multi_dot, outer, trace, Universal Functions (ufunc), all, any, around, round_, fix, [nan][cum]prod, [nan][cum]sum, diff, ediff1d, gradient, cross, trapz, amax, nanmax, amin, nanmin, clip, random.Generator.random, random.Generator.permuted, random.Generator.standard_[exponential|gamma|normal], some ma and ma.MaskedArray methods
        case ["numpy", _] if "out" in kwargs and kwargs["out"] is not None:
            return {"returns": 0, "out": INOUT}

        case ["numpy", "split"]:
            raise NotImplementedError()

        case ["numpy", "linalg", "qr"]:
            mode = kwargs.get("mode", "reduced")

            match mode:
                case "complete" | "reduced" | "raw":
                    return {"returns": 2}
                case "r":
                    return {"returns": 1}
                case _:
                    raise ValueError(f"{qualname}(..., mode='{mode}') is not valid")

        case ["numpy", "linalg", "svd"]:
            compute_uv = kwargs.get("compute_uv", True)
            return {"returns": 3 if compute_uv else 1}

        case ["numpy", "linalg", "eig" | "eigh"]:
            return {"returns": 2}

        case ["numpy", "linalg", "lstsq"]:
            return {"returns": 4}

        case ["numpy", "random", _, "shuffle"]:
            raise NotImplementedError(f"'{qualname}' acts in-place but COMPSs direction cannot be mark correctly as it is a built-in method. Please use 'numpy.random.Generator.permuted(..., out=OUT)' instead.")

        case ["numpy", _]:
            return {"returns": 1}

    return {}
