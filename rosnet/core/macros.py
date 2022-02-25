from typing import Callable
from functools import wraps
from multimethod import multimethod
from autoray import register_function

# from https://github.com/dask/dask/blob/95fb60a31a87c6b94b01ed75ab6533fa04d51f19/dask/utils.py
def inherit_doc(parent):
    def wrapper(method):
        extra = getattr(method, "__doc__", None) or ""
        # parent = getattr(lib, method.__name__)
        method.__doc__ = parent.__doc__ or ""
        method.__doc__ += "\n\n" + extra.rstrip("\n")
        return method

    return wrapper


@multimethod
def implements(function: str, ext=None):
    "Register a function (e.g. __array_function__) implementation."

    def registrar(func):
        backend = f"rosnet.{ext}" if ext else "rosnet"
        register_function(backend, function, func)
        return func

    return registrar


@implements.register
def implements(function: Callable, ext=None):
    "Register a function (e.g. __array_function__) implementation."

    def registrar(func):
        backend = f"rosnet.{ext}" if ext else "rosnet"
        register_function(backend, function.__name__, func)
        inherit_doc(function)(func)
        return func

    return registrar


def todo(func):
    @wraps(func)
    def todo_msg(*args, **kwargs):
        raise NotImplementedError(f"{func.__name__} is not implemented yet")

    return todo_msg
