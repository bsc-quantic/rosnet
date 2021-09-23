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


def tensordot_sequential(a: List, b: List, axes):
    # pylint: disable=protected-access
    import rosnet

    blockshape = result_shape(a[0].shape, b[0].shape, axes)
    dtype = np.result_type(
        *(i.dtype if hasattr(i, "dtype") else i for i in chain(a, b))
    )

    # get ref if a,b are COMPSsArrays
    aref = [i._ref if hasattr(i, "_ref") else i for i in a]
    bref = [i._ref if hasattr(i, "_ref") else i for i in b]

    ref = rosnet.task.tensordot.sequential(aref, bref, axes)
    return rosnet.COMPSsArray(ref, shape=blockshape, dtype=dtype)


def tensordot_commutative(a: List, b: List, axes):
    # pylint: disable=protected-access
    import rosnet

    blockshape = result_shape(a[0].shape, b[0].shape, axes)
    dtype = np.result_type(
        *(i.dtype if hasattr(i, "dtype") else i for i in chain(a, b))
    )

    buffer = rosnet.task.tensordot.ArrayWrapper()

    for ia, ib in zip(a, b):
        # get ref if a,b are COMPSsArrays
        ia = ia._ref if hasattr(ia, "_ref") else ia
        ib = ib._ref if hasattr(ib, "_ref") else ib

        rosnet.task.tensordot.commutative(buffer, ia, ib, axes)

    return rosnet.COMPSsArray(buffer, shape=blockshape, dtype=dtype)


def tensordot_tensordot(a: List, b: List, axes):
    # pylint: disable=protected-access
    import rosnet

    if len(a) > 1 or len(b) > 1:
        raise ValueError()

    blockshape = result_shape(a[0].shape, b[0].shape, axes)
    dtype = np.result_type(
        *(i.dtype if hasattr(i, "dtype") else i for i in chain(a, b))
    )

    # get ref if a,b are COMPSsArrays
    aref = [i._ref if hasattr(i, "_ref") else i for i in a][0]
    bref = [i._ref if hasattr(i, "_ref") else i for i in b][0]

    ref = rosnet.task.tensordot.tensordot(aref, bref, axes)
    return rosnet.COMPSsArray(ref, shape=blockshape, dtype=dtype)
