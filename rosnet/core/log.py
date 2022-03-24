from typing import Callable
import functools
import logging
import sys
import inspect

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


class ArgumentLog:
    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG):
        if not isinstance(logger, logging.Logger):
            raise ValueError(f"logging.Logger must be passed but {type(logger)} instead")

        self.logger = logger
        self.level = level

    def __call__(self, f: Callable) -> Callable:
        spec = inspect.getfullargspec(f)
        if spec.varargs is not None:
            raise NotImplementedError("varargs not supported in Log decorator yet")

        msg = str.join(", ", ("%s={}" % arg for arg in spec.args))

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            try:
                ret = f(*args, **kwargs)
            finally:
                # TODO use Log.level
                logger.debug(msg.format(*args))

        return wrapper


log_args = ArgumentLog
