# from https://github.com/dask/dask/blob/95fb60a31a87c6b94b01ed75ab6533fa04d51f19/dask/utils.py
def inherit_doc(parent):
    def wrapper(method):
        extra = getattr(method, "__doc__", None) or ""
        # parent = getattr(lib, method.__name__)
        method.__doc__ = parent.__doc__ or ""
        method.__doc__ += "\n\n" + extra.rstrip("\n")
        return method

    return wrapper


numpy_dispatcher = {}


def implements(np_function, cls):
    "Register an __array_function__ implementation."
    # pylint: disable=protected-access

    def registrar(func):
        if cls not in numpy_dispatcher:
            numpy_dispatcher.update({cls: {}})

        numpy_dispatcher[cls].update({np_function: func})
        inherit_doc(np_function)(func)
        return func

    return registrar


def isarray(o: object):
    return (
        isinstance(o, np.ndarray)
        or issubclass(o, np.ndarray)
        or hasattr(o, "__array__")
    )