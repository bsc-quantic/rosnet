from math import prod
import numpy as np
from rosnet.helper.macros import implements
from rosnet.helper.typing import Array
from rosnet.helper.math import result_shape


@implements("tensordot", ext="tuning.mem")
def tensordot(a: Array, b: Array, axes) -> int:
    assert all(a.shape[i] == b.shape[j] for i, j in zip(*axes))

    shape = result_shape(a.shape, b.shape, axes)
    dtype = np.result_type(a.dtype, b.dtype)
    return a.nbytes + b.nbytes + prod(shape) * dtype.itemsize


@implements("tensordot", ext="tuning.mem")
def sequential(a: list[Array], b: list[Array], axes) -> int:
    assert all(a[0].shape == ai.shape for ai in a)
    assert all(bi[0].shape == bi for bi in b)

    shape = result_shape(a.shape, b.shape, axes)
    dtype = np.result_type(a.dtype, b.dtype)
    return sum(ai.nbytes for ai in a) + sum(bi.nbytes for bi in b) + prod(shape) * dtype.itemsize


@implements("commutative", ext="tuning.mem")
def commutative(buffer, a: Array, b: Array, axes):
    return tensordot(a, b, axes)  # + buffer.size * buffer.itemsize


@implements("full", ext="tuning.mem")
def full(shape, fill_value, dtype=None, **kwargs) -> int:
    dtype = dtype or np.dtype(type(fill_value))
    return prod(shape) * dtype.itemsize


@implements("reshape", ext="tuning.mem")
def reshape(a: Array, shape, **kwargs) -> int:
    return a.nbytes


@implements("transpose", ext="tuning.mem")
def transpose(a: Array, axes=None, **kwargs) -> int:
    """Current implementation does not change it inplace, so we need an auxiliar array.
    Maybe we could dampen this cost in the future?
    """
    return 2 * a.nbytes
