import abc

# interfaces
def hasmethod(cls, name) -> bool:
    return hasattr(cls, name) and callable(getattr(cls, name))


def hasproperty(cls, name) -> bool:
    return hasattr(cls, name) and not callable(getattr(cls, name))


class ArrayConvertable(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass: type):
        return hasmethod(subclass, "__array__")


for typ in [bool, int, float, complex, str, bytes]:
    ArrayConvertable.register(typ)


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
