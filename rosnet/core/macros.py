from functools import wraps

# from https://github.com/dask/dask/blob/95fb60a31a87c6b94b01ed75ab6533fa04d51f19/dask/utils.py
def inherit_doc(parent):
    def wrapper(method):
        extra = getattr(method, "__doc__", None) or ""
        # parent = getattr(lib, method.__name__)
        method.__doc__ = parent.__doc__ or ""
        method.__doc__ += "\n\n" + extra.rstrip("\n")
        return method

    return wrapper


def todo(func):
    @wraps(func)
    def todo_msg(*args, **kwargs):
        raise NotImplementedError(f"{func.__name__} is not implemented yet")

    return todo_msg
