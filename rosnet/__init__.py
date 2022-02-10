__version__ = "0.2"
from rosnet.array import COMPSsArray, BlockArray, full, ones, zeros, rand, array

# opt-einsum support
from rosnet.array import __compss_tensordot, __block_tensordot


def tensordot(a, b, axes):
    if isinstance(a, COMPSsArray) and isinstance(b, COMPSsArray):
        return __compss_tensordot(a, b, axes)
    elif isinstance(a, BlockArray) and isinstance(b, BlockArray):
        return __block_tensordot(a, b, axes)
    else:
        return NotImplemented
