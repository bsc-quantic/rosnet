import abc
import numpy as np
from multimethod import multimethod
from rosnet.helper.macros import implements

# numpy
@implements(np.tensordot)
@multimethod
def tensordot(*args, **kwargs):
    raise NotImplementedError()


@implements(np.einsum)
@multimethod
def einsum(*args, **kwargs):
    raise NotImplementedError()


@implements(np.reshape)
@multimethod
def reshape(*args, **kwargs):
    raise NotImplementedError()


@implements(np.transpose)
@multimethod
def transpose(*args, **kwargs):
    raise NotImplementedError()


@implements(np.stack)
@multimethod
def stack(*args, **kwargs):
    raise NotImplementedError()


@implements(np.split)
@multimethod
def split(*args, **kwargs):
    raise NotImplementedError()


@implements(np.block)
@multimethod
def block(*args, **kwargs):
    raise NotImplementedError()


@implements(np.zeros_like)
@multimethod
def zeros_like(*args, **kwargs):
    raise NotImplementedError()


@implements(np.ones_like)
@multimethod
def ones_like(*args, **kwargs):
    raise NotImplementedError()


@implements(np.full_like)
@multimethod
def full_like(*args, **kwargs):
    raise NotImplementedError()


@implements(np.empty_like)
@multimethod
def empty_like(*args, **kwargs):
    raise NotImplementedError()


# custom
@multimethod
def to_numpy(*args, **kwargs):
    raise NotImplementedError()


# interfaces
def hasmethod(cls, name) -> bool:
    return hasattr(cls, name) and callable(getattr(cls, name))


def hasproperty(cls, name) -> bool:
    return hasattr(cls, name) and not callable(getattr(cls, name))


class ArrayConvertable(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass: type):
        return hasmethod(subclass, "__array__")


class ArrayDispatchable(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass: type):
        return hasmethod(subclass, "__array_ufunc__") and hasmethod(subclass, "__array_function__")


class Array(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass: type):
        return issubclass(subclass, ArrayConvertable) and issubclass(subclass, ArrayDispatchable) and hasproperty(subclass, "shape") and hasproperty(subclass, "dtype")


class Future(metaclass=abc.ABCMeta):
    pass


try:
    from pycompss.runtime.management.classes import Future as COMPSsFuture

    Future.register(COMPSsFuture)
except:
    pass
