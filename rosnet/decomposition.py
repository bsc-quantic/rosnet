from rosnet.utils import isunique, prod, ispower2
from itertools import chain
import numpy as np
import dislib as ds
from dislib.data.array import Array
from typing import Tuple

__all__ = ['schmidt']


def schmidt(a: Tensor, axes_v: Tuple[int], chi=None, eps=1e-9) -> (Tensor, Tensor):
    """ Decomposes tensor `a` into two tensors `u` and `v` using the Schmidt decomposition.

    `a`: `Tensor` to decompose.

    `axes_v`: `tuple`-like. List of indexes kept by `v`.

    `chi`: `int`. Maximum rank of the decomposition. Currently unused.

    `eps`: `float`. Epsilon.
    """
    if not isinstance(axes_v, list) and not isinstance(axes_v, tuple):
        raise TypeError(
            f'axes_v must be of type "tuple" or "list"; {type(axes)} instead')
    if not isunique(axes_v) or len(axes_v) > a.rank - 1 or min(axes_v) < 0 or max(axes_v) >= a.rank:
        raise ValueError(f'axes_v is not valid: {axes_v}')
    if not isinstance(chi, int):
        raise TypeError(f'chi must be of type "int"; {type(chi)} instead')
    if chi < 2 or not ispower2(chi):
        raise ValueError(f'chi must be a power of 2 and > 2: {chi}')

    axes_u = tuple(filter(lambda x: x not in axes_v, range(a.rank)))

    m = prod(a.shape[i] for i in axes_u)
    n = prod(a.shape[i] for i in axes_v)
    mb = prod(a.block_shape[i] for i in axes_u)
    nb = prod(a.block_shape[i] for i in axes_v)
    k, kb = min(m, n), min(mb, nb)

    shape_u = [a.shape[i] for i in axes_u] + [k]
    shape_v = [a.shape[i] for i in axes_v] + [k]
    bshape_u = [a.block_shape[i] for i in axes_u] + [k]
    bshape_v = [a.block_shape[i] for i in axes_v] + [k]

    # permute tensor
    permutation = tuple(chain(axes_u, axes_v))
    a.transpose(permutation)

    # reshape to matrix
    a.reshape((m, n), (mb, np))

    # perform SVD
    U, V = svd(a, chi, eps)

    # reshape U, V to tensors
    U.reshape(shape_u, bshape_u)
    V.reshape(shape_v, bshape_v)

    return (U, V)


def svd(A: Tensor, k, eps) -> (Tensor, Tensor):
    """ Computes the Singular Value Decomposition of `A`.

    `A`: rank-2 `Tensor`.

    `k`: int. Currently unused.

    `eps`: Epsilon.
    """
    if A.rank != 2:
        raise ValueError("A must be a matrix!")
    if k > min(A.shape):
        raise ValueError("")
    if k % A._block_shape[1] != 0:
        raise NotImplementedError(
            "reduced rank must be a multiple of block_shape[1]")

    m, n = A.shape
    k = min(m, n)
    mb, nb = A.block_shape
    kb = min(mb, nb)

    # call SVD
    dsA = Array(A._blocks.tolist(), U.block_shape,
                U.block_shape, U.shape, False, delete=False)
    dsU, dsS, dsV = ds.svd(dsA, eps=eps)

    # NOTE hack not to delete blocks when moved to U,V
    dsU._delete = False
    dsV._delete = False

    # transform back U,V to Tensor
    U = Tensor(np.ndarray(dsU._blocks), dsU.shape, dsU._reg_shape)
    V = Tensor(np.ndarray(dsV._blocks), dsV.shape, dsV._reg_shape)

    return (U, V)

