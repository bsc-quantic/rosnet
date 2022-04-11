import sys
from typing import Callable, TypeVar

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

import functools
import inspect
import logging
from contextlib import contextmanager

from multimethod import multimethod

T = TypeVar("T")
P = ParamSpec("P")

logging.getLogger("rosnet").addHandler(logging.NullHandler())

if __debug__:
    logger = logging.getLogger("rosnet")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers() or all(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        formatter = logging.Formatter("[%(name)-12s][%(levelname)-8s] @ %(funcName)-20s: %(message)s")

        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)

        logger.addHandler(handler)


@multimethod
def trace(*args, **kwargs):
    raise NotImplementedError()


@trace.register()  # default
def trace_in_stderr(f: Callable[P, T]) -> Callable[P, T]:
    if not __debug__:
        return f

    signature = inspect.signature(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs) -> T:
        ret = None
        try:
            sys.stdout.write(f"[rosnet] args={args}, kwargs={kwargs}\n")
            sys.stdout.flush()
            ret = f(*args, **kwargs)
            return ret

        finally:
            ba = signature.bind(*args, **kwargs)
            ba.apply_defaults()

            msg = str.join(", ", (f"{arg}={value}" for arg, value in ba.arguments.items()))
            if ret is not None:
                msg += f" -> {ret}"

            sys.stdout.write(f"[rosnet][{f.__name__}] @ {msg}\n")
            sys.stdout.flush()

    return wrapper


@trace.register
def trace_in_logger(logger: logging.Logger, /, level: int = logging.DEBUG):
    return ArgumentLog(logger, level=level)


class ArgumentLog:
    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG):
        if not isinstance(logger, logging.Logger):
            raise ValueError(f"logging.Logger must be passed but {type(logger)} instead")

        self.logger = logger
        self.level = level

    @contextmanager
    def _fix_funcName(self, fn: str):
        store = {}
        formatter = logging.Formatter(f"[%(name)-12s][%(levelname)-8s] @ {fn:20}: %(message)s")
        logger = logging.getLogger("rosnet")

        for i, handler in filter(lambda x: not isinstance(x[1], logging.NullHandler), enumerate(logger.handlers)):
            store[i] = handler.formatter
            handler.formatter = formatter

        yield

        for i, fmt in store.items():
            logger.handlers[i].formatter = fmt

    def __call__(self, f: Callable[P, T]) -> Callable[P, T]:
        signature = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> T:
            ret = None
            try:
                ret = f(*args, **kwargs)
                return ret

            finally:
                ba = signature.bind(*args, **kwargs)
                ba.apply_defaults()

                msg = str.join(", ", (f"{arg}={value}" for arg, value in ba.arguments.items()))
                if ret is not None:
                    msg += f" -> {ret}"

                with self._fix_funcName(f.__name__):
                    self.logger.log(self.level, msg)

        return wrapper


log_args = ArgumentLog
