import operator as op
from types import GetSetDescriptorType
import typing
from typing import Generic, TypeVar, Callable, Union
import sys
import itertools
import inspect

if sys.version_info >= (3, 10):
    from typing import ParamSpec, Self
else:
    from typing_extensions import ParamSpec, Self

import pytypes
import numpy as np

T = TypeVar("T")
P = ParamSpec("P")


# TODO automatic return type inference (i.e. override __new__? use pytypes)
class Deferred(Generic[T]):
    """A deferred execution."""

    def __init__(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    # @property
    # def sign(self):
    #     return inspect.signature(self.func)

    @property
    def type(self) -> type:
        try:
            return typing.get_args(pytypes.get_orig_class(self))[0]
        except AttributeError:
            return infer_return_type(self.func, *self.args, **self.kwargs)

    def deps(self) -> "tuple[Self, ...]":
        """Lists all `Deferred` dependencies.

        ### Returns
        tuple[Deferred,...]
            List of dependencies.
        """
        return tuple(filter(lambda x: isinstance(x, Deferred), itertools.chain(self.args, self.kwargs)))

    def __getattr__(self, name: str) -> Union[Callable[P, "Deferred"], "Deferred"]:
        """Defers instance method if type can be inferred.

        ### Parameters
        name: str
            Name of the attribute or instance method.

        ### Return
        Union[Callable[P, Deferred], Deferred]
        """
        if self.type is None:
            raise AttributeError("can't infer return type or None")

        attr = getattr(self.type, name)
        if isinstance(attr, Callable):
            return defer(attr)
        elif isinstance(attr, property):
            return defer(attr.fget)(self)
        elif isinstance(attr, GetSetDescriptorType):
            # defined in CPython with PyGetSetDef (e.g. some numpy "properties")
            return defer(attr.__get__)(self)
        else:
            raise AttributeError()

    def __abs__(self) -> Self:
        return Deferred(op.abs, self)  # type: ignore

    def __add__(self, other) -> Self:
        return Deferred(op.add, self, other)

    def __radd__(self, other) -> Self:
        return Deferred(op.add, other, self)

    def __sub__(self, other) -> Self:
        return Deferred(op.sub, self, other)

    def __rsub__(self, other) -> Self:
        return Deferred(op.sub, other, self)

    def __mul__(self, other) -> Self:
        return Deferred(op.mul, self, other)

    def __rmul__(self, other) -> Self:
        return Deferred(op.mul, other, self)

    def __matmul__(self, other) -> Self:
        return Deferred(op.matmul, self, other)

    def __rmatmul__(self, other) -> Self:
        return Deferred(op.matmul, other, self)

    def __truediv__(self, other) -> Self:
        return Deferred(op.truediv, self, other)

    def __rtruediv__(self, other) -> Self:
        return Deferred(op.truediv, other, self)

    def __floordiv__(self, other) -> Self:
        return Deferred(op.floordiv, self, other)

    def __rfloordiv__(self, other) -> Self:
        return Deferred(op.floordiv, other, self)

    def __mod__(self, other) -> Self:
        return Deferred(op.mod, self, other)

    def __rmod__(self, other) -> Self:
        return Deferred(op.mod, other, self)

    def __divmod__(self, other):
        return Deferred(divmod, self, other)  # type: ignore

    def __rdivmod__(self, other):
        return Deferred(divmod, other, self)  # type: ignore

    def __pow__(self, other):
        return Deferred(op.pow, self, other)

    def __rpow__(self, other):
        return Deferred(op.pow, other, self)

    def __lshift__(self, other):
        return Deferred(op.lshift, self, other)

    def __rlshift__(self, other):
        return Deferred(op.lshift, other, self)

    def __rshift__(self, other):
        return Deferred(op.rshift, self, other)

    def __rrshift__(self, other):
        return Deferred(op.rshift, other, self)

    def __and__(self, other):
        return Deferred(op.and_, self, other)

    def __rand__(self, other):
        return Deferred(op.and_, other, self)

    def __xor__(self, other):
        return Deferred(op.xor, self, other)

    def __rxor__(self, other):
        return Deferred(op.xor, other, self)

    def __or__(self, other):
        return Deferred(op.or_, self, other)

    def __ror__(self, other):
        return Deferred(op.or_, other, self)


def defer(func: Callable[P, T]) -> Callable[P, Deferred]:
    def call_defer(*args: P.args, **kwargs: P.kwargs) -> Deferred:
        return Deferred(func, *args, **kwargs)

    return call_defer


def infer_return_type(func, *args, **kwargs) -> type:
    # TODO more cases: non-numpy classes
    if inspect.isbuiltin(func):
        raise NotImplementedError()
    elif "dtype" in kwargs:
        dtype = kwargs["dtype"]
        if isinstance(dtype, np.dtype):
            return dtype.type
        raise NotImplementedError()
    elif "out" in kwargs:
        raise NotImplementedError()
    elif isinstance(func, np.ufunc):
        # TODO `out` kwarg case, `casting` kwarg, `dtype` kwarg...
        return np.result_type(*args).type
    else:
        typ = np.result_type(*args).type
        if not isinstance(typ, np.object_):
            return typ

        raise NotImplementedError()


__all__ = ["defer", "Deferred"]
