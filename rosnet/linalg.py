import copy
from rosnet import Tensor, kernel
import numpy as np
from itertools import product
from pycompss.api.api import TaskGroup


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


# TODO implement reduced SVD variants
def svd(A: Tensor, eps=1e-9, copy=True) -> (Tensor, Tensor):
    """ Computes the Singular Value Decomposition of `A`.

    `A`: rank-2 `Tensor`.

    `eps`: Epsilon.
    """
    if A.rank != 2:
        raise ValueError("A must be a matrix!")

    m, n = A.shape
    k = min(m, n)
    mb, nb = A.block_shape
    kb = min(mb, nb)

    # generate U,V
    U = copy.deepcopy(A) if copy else A  # TODO is U.block_shape ok?
    V = identity(n, (nb, nb))

    # call SVD asynchronously
    kernel.svdmatrix_async_blocked(U._blocks.tolist(), V._blocks.tolist(), eps)

    return (U, V)


def identity(n, block_shape, dtype=None) -> Tensor:
    if len(block_shape) != 2:
        raise ValueError("block_shape needs to have 2 values")
    if n < block_shape[0] or n < block_shape[1]:
        raise ValueError("block is greater than the tensor")

    grid = tuple(n // bs for bs in block_shape)
    tensorid = str(next(Tensor._newid))
    with TaskGroup(tensorid, False):
        blocks = [kernel.block_identity(block_shape, n, i, j, dtype)
                  for i, j in product(range(g) for g in grid)]

    # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
    if isinstance(blocks[0], np.ndarray):
        bs = np.empty_like(range(len(blocks)), dtype=np.ndarray)
        for i, _ in enumerate(blocks):
            bs[i] = blocks[i]
        blocks = bs.reshape(grid)
    else:
        blocks = np.array(blocks).reshape(grid)
    return Tensor(blocks, (n, n), block_shape, True, tensorid)
