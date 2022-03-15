from typing import Literal
import inspect
import numpy as np
from rosnet import dispatch

EXPLICITLY_DISPATCH = [
    np.zeros,
    np.ones,
    np.full,
    np.random.rand,
]


class ArrayFunctionMixin:
    def __array_function__(self, func, types, args, kwargs):
        """Dispatches specialized implementation using multimethod.

        NOTES
        -----
        For numpy functions zeros, ones, full, random.rand and other functions defined in EXPLICITLY_DISPATCH, multimethod has no hint of the type in the arguments. We give two solutions for this problem:
        - Prepend a `Literal[cls]` object to the positional arguments list.
        - Inspect the module of the class and match the function name.
        """
        if func in EXPLICITLY_DISPATCH:
            cls = type(self)
            module = inspect.getmodule(cls)

            if hasattr(dispatch, func.__name__):
                return getattr(dispatch, func.__name__)(Literal[cls], *args, **kwargs)
            elif hasattr(module, func.__name__):
                return getattr(module, func.__name__)(*args, **kwargs)
            else:
                return NotImplemented

        module = dispatch
        if inspect.getmodule(func) == np.linalg:
            module = dispatch.linalg

        return getattr(module, func.__name__)(*args, **kwargs) if hasattr(module, func.__name__) else NotImplemented
