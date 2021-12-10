__version__ = "0.1"
from rosnet.array import COMPSsArray, BlockArray, full, ones, zeros, rand, array


def tensordot(a, b, axes):
    import numpy as np

    if isinstance(a, COMPSsArray) and isinstance(b, COMPSsArray):
        return np.tensordot(a, b, axes)
    elif isinstance(a, BlockArray) and isinstance(b, BlockArray):
        return np.tensordot(a, b, axes)
    else:
        return NotImplemented
