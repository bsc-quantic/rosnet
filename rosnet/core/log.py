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
        signature = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
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

                logger.log(self.level, msg)

        return wrapper


log_args = ArgumentLog
