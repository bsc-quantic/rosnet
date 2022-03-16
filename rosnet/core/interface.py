import abc
import importlib.util
from typing import TYPE_CHECKING, Union

# interfaces
def hasmethod(cls, name) -> bool:
    return hasattr(cls, name) and callable(getattr(cls, name))


def hasproperty(cls, name) -> bool:
    return hasattr(cls, name) and not callable(getattr(cls, name))


class ArrayConvertable(metaclass=abc.ABCMeta):
    """Formal interface for classes that can be transformed into numpy arrays.
    Implements the ArrayLike protocol from numpy.
    """

    @classmethod
    def __subclasshook__(cls, subclass: type):
        return hasmethod(subclass, "__array__")


for typ in [bool, int, float, complex, str, bytes]:
    ArrayConvertable.register(typ)


class ArrayDispatchable(metaclass=abc.ABCMeta):
    """Formal interface for classes that implement the numpy dispatch mechanism."""

    @classmethod
    def __subclasshook__(cls, subclass: type):
        return hasmethod(subclass, "__array_ufunc__") and hasmethod(subclass, "__array_function__")


class Array(metaclass=abc.ABCMeta):
    """Formal interface for classes that fulfill the ArrayConvertable and ArrayDispatchable interfaces."""

    if TYPE_CHECKING:

        def __return_bool(self, *args, **kwargs) -> bool:  # type: ignore[type-var]
            pass

        __lt__ = __return_bool
        __le__ = __return_bool
        __eq__ = __return_bool
        __ne__ = __return_bool
        __gt__ = __return_bool
        __ge__ = __return_bool

        def __return_array(self, *args, **kwargs) -> Array:  # type: ignore
            pass

        __add__ = __radd__ = __iadd__ = __return_array
        __sub__ = __rsub__ = __isub__ = __return_array
        __mul__ = __rmul__ = __imul__ = __return_array
        __matmul__ = __rmatmul__ = __imatmul__ = __return_array
        __truediv__ = __rtruediv__ = __itruediv__ = __return_array
        __floordiv__ = __rfloordiv__ = __ifloordiv__ = __return_array
        __mod__ = __rmod__ = __imod__ = __return_array
        __divmod__ = __rdivmod__ = __return_array
        __pow__ = __rpow__ = __ipow__ = __return_array
        __lshift__ = __rlshift__ = __ilshift__ = __return_array
        __rshift__ = __rrshift__ = __irshift__ = __return_array
        __and__ = __rand__ = __iand__ = __return_array
        __xor__ = __rxor__ = __ixor__ = __return_array
        __or__ = __ror__ = __ior__ = __return_array

        __neg__ = __return_array
        __pos__ = __return_array
        __abs__ = __return_array
        __invert__ = __return_array

    @classmethod
    def __subclasshook__(cls, subclass: type):
        return issubclass(subclass, ArrayConvertable) and issubclass(subclass, ArrayDispatchable) and hasproperty(subclass, "shape") and hasproperty(subclass, "dtype")


if TYPE_CHECKING:
    import numpy as np

    for typ in [bool, int, float, complex, str, bytes, np.generic]:
        Array.register(typ)


class Future(metaclass=abc.ABCMeta):
    pass


try:
    from pycompss.runtime.management.classes import Future as COMPSsFuture

    Future.register(COMPSsFuture)
except:
    pass


class AsyncArray(metaclass=abc.ABCMeta):
    data: Union[Array, Future]
