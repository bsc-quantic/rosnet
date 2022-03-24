import logging
import sys

logging.getLogger("rosnet").addHandler(logging.NullHandler())

if __debug__:
    logger = logging.getLogger("rosnet")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers() or all(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setLevel(logging.DEBUG)

        logger.addHandler(handler)
