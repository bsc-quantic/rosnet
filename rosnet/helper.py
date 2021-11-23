from typing import List
from itertools import chain
import numpy as np
from .utils import result_shape, inherit_doc, prod

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
