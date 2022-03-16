from typing import Sequence
from math import prod
import numpy as np
from rosnet.core.interface import Array
from rosnet.core.math import result_shape


def tensordot(a: Array, b: Array, axes) -> int:
    assert all(a.shape[i] == b.shape[j] for i, j in zip(*axes))

    shape = result_shape(a.shape, b.shape, axes)
    dtype = np.result_type(a.dtype, b.dtype)
    return a.nbytes + b.nbytes + prod(shape) * dtype.itemsize


def sequential(a: Sequence[Array], b: Sequence[Array], axes) -> int:
    assert all(a[0].shape == ai.shape for ai in a)
    assert all(bi[0].shape == bi for bi in b)

    shape = result_shape(a.shape, b.shape, axes)
    dtype = np.result_type(a.dtype, b.dtype)
    return sum(ai.nbytes for ai in a) + sum(bi.nbytes for bi in b) + prod(shape) * dtype.itemsize


def commutative(buffer, a: Array, b: Array, axes):
    return tensordot(a, b, axes)  # + buffer.size * buffer.itemsize


def full(shape, fill_value, dtype=None, **kwargs) -> int:
    dtype = dtype or np.dtype(type(fill_value))
    return prod(shape) * dtype.itemsize


def reshape(a: Array, shape, **kwargs) -> int:
    return a.nbytes


def transpose(a: Array, axes=None, **kwargs) -> int:
    """Current implementation does not change it inplace, so we need an auxiliar array.
    Maybe we could dampen this cost in the future?
    """
    return 2 * a.nbytes


def qr_complete(a: Array) -> int:
    n = a.shape[-1]
    m = a.shape[-2]
    rest = a.shape[0:-2]
    return a.nbytes + a.itemsize * prod((*rest, m, m)) + a.itemsize * prod((*rest, m, n))


def qr_reduced(a: Array) -> int:
    n = a.shape[-1]
    m = a.shape[-2]
    k = min(m, n)
    rest = a.shape[0:-2]
    return a.nbytes + a.itemsize * prod((*rest, m, k)) + a.itemsize * prod((*rest, k, n))


def qr_r(a: Array) -> int:
    n = a.shape[-1]
    m = a.shape[-2]
    k = min(m, n)
    rest = a.shape[0:-2]
    return a.nbytes + a.itemsize * prod((*rest, k, n))


def qr_raw(a: Array) -> int:
    n = a.shape[-1]
    m = a.shape[-2]
    k = min(m, n)
    rest = a.shape[0:-2]
    return a.nbytes + a.itemsize * prod((*rest, n, m)) + a.itemsize * prod((*rest, k))
