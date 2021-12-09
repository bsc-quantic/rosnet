from copy import deepcopy
from rosnet import Tensor, kernel
from rosnet.utils import ndarray_from_list
import numpy as np
from itertools import product
from pycompss.api.api import TaskGroup


def gemm(a: Tensor, b: Tensor, transpose_a: bool, transpose_b: bool) -> Tensor:
    if a.rank != 2 or b.rank != 2:
        raise ValueError("a,b must be matrixes")

    mi = 1 if transpose_a else 0
    ni = 0 if transpose_b else 1

    shape = (a.shape[mi], b.shape[ni])
    block_shape = (a.block_shape[mi], b.block_shape[ni])
    grid = [s // bs for s, bs in zip(shape, block_shape)]
    blocks = np.empty(grid, dtype=object)

    tensorid = str(next(Tensor._newid))
    with TaskGroup(tensorid, False), np.nditer(blocks, flags=["multi_index"], op_flags=["writeonly"]) as it:
        for block in it:
            row, col = tuple(it)
            blocks_a = a._blocks[:, row] if transpose_a else a._blocks[row, :]
            blocks_b = b._blocks[col, :] if transpose_b else b._blocks[:, col]
            block[...] = kernel.block_gemm(blocks_a, blocks_b, transpose_a, transpose_b)

    return Tensor(blocks, shape, block_shape, True, tensorid)


# TODO implement reduced SVD variants
def svd(A: Tensor, eps=1e-9, copy=True, parallel=False) -> (Tensor, Tensor):
    """Computes the Singular Value Decomposition of `A`.

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
    U = deepcopy(A) if copy else A  # TODO is U.block_shape ok?
    V = identity(n, (nb, nb))

    # call SVD asynchronously
    if parallel:
        kernel.svdmatrix_async_nested(U._blocks.tolist(), V._blocks.tolist(), eps)
    else:
        kernel.svdmatrix_async(U._blocks.tolist(), V._blocks.tolist())

    return (U, V)


def identity(n, block_shape, dtype=None) -> Tensor:
    if len(block_shape) != 2:
        raise ValueError("block_shape needs to have 2 values")
    if n < block_shape[0] or n < block_shape[1]:
        raise ValueError("block is greater than the tensor")

    grid = tuple(n // bs for bs in block_shape)
    tensorid = str(next(Tensor._newid))
    with TaskGroup(tensorid, False):
        blocks = [kernel.block_identity(block_shape, n, i, j, dtype) for i, j in product(range(grid[0]), range(grid[1]))]

    blocks = ndarray_from_list(blocks, grid)
    return Tensor(blocks, (n, n), block_shape, True, tensorid)
