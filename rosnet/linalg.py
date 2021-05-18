import dislib as ds
from dislib.data.array import Array
from pycompss.api.task import task
from pycompss.api.parameter import Type, Depth, IN, INOUT, COLLECTION_INOUT, COLLECTION_IN, COLLECTION_OUT
from rosnet import Tensor
import numpy as np
from itertools import product


def gemm(a: Tensor, b: Tensor, axes) -> Tensor:
    # TODO perform permutation
    a.transpose(...)
    b.transpose(...)

    a.reshape(...)
    b.reshape(...)

    # perform matrix multiplication
    A = Array(a._blocks.tolist(), a.block_shape,
              a.block_shape, a.shape, False, delete=False)
    B = Array(b._blocks.tolist(), b.block_shape,
              b.block_shape, b.shape, False, delete=False)
    C = A @ B

    c = Tensor(np.array(C._blocks), ..., ...)
    c.reshape(...)
    return c


def svd(A: Array, k, eps) -> (Array, Array):
    """ Computes the Singular Value Decomposition of `A`.

    `A`: rank-2 `Array`.

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
    dsA = Array(A._blocks.tolist(), A.block_shape,
                A.block_shape, A.shape, False, delete=False)
    dsU, dsS, dsV = ds.svd(dsA, eps=eps)

    # TODO contract dsU and dsS

    # NOTE hack not to delete blocks when moved to U,V
    dsU._delete = False
    dsV._delete = False

    # transform back U,V to Array
    # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
    blocks = dsU._blocks
    if isinstance(blocks[0][0], np.ndarray):
        bs = np.empty(
            (len(blocks), len(blocks[0])), dtype=np.ndarray, order='F')
        for i, j in product(range(len(blocks)), range(len(blocks[0]))):
            bs[i, j] = blocks[i][j]
        blocks = bs
    U = Array(blocks, list(dsU.shape), list(dsU._reg_shape))

    blocks = dsV._blocks
    if isinstance(blocks[0][0], np.ndarray):
        bs = np.empty(
            (len(blocks), len(blocks[0])), dtype=np.ndarray, order='F')
        for i, j in product(range(len(blocks)), range(len(blocks[0]))):
            bs[i, j] = blocks[i][j]
        blocks = bs
    V = Array(blocks, list(dsV.shape), list(dsV._reg_shape))

    return (U, V)


@task(A={Type: COLLECTION_INOUT, Depth: 1}, U={Type: COLLECTION_INOUT, Depth: 1}, V={Type: COLLECTION_INOUT, Depth: 1})
def svd_serial(A, U, S, V, m, n):
    pass
