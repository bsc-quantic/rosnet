import os
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import Type, Depth, IN, INOUT, COLLECTION_INOUT, COLLECTION_IN, COLLECTION_OUT
from pycompss.api.api import compss_wait_on
import itertools


def __numpy_transpose(a, axes=None):
    return np.asfortranarray(np.transpose(a, axes))


# defaults
__backend_transpose = os.environ.get('ROSNET_BACKEND_TRANSPOSE', 'numpy')

if __backend_transpose == 'numpy':
    transpose = __numpy_transpose
else:
    try:
        transpose = __import__(__backend_transpose,
                               fromlist=('transpose')).transposes
    except ImportError:
        transpose = __numpy_transpose
        __backend_transpose = 'numpy'


@task(block=INOUT, returns=np.ndarray)
def block_transpose(block: np.ndarray, permutator):
    block[...] = transpose(block, permutator)


@task(block=INOUT, returns=np.ndarray)
def block_reshape(block: np.ndarray, shape: tuple):
    block[...] = block.reshape(shape, order='F')


@task(block=IN, returns={Type: COLLECTION_OUT, Depth: 1})
def block_split(block: np.ndarray, n: int, axis: int):
    return map(lambda x: x.copy(), np.split(block, n, axis))


@task(blocks=COLLECTION_IN, returns=np.ndarray)
def block_merge(blocks, axis: int):
    return np.stack(blocks, axis)


@task(returns=np.array)
def block_full(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)


@task(returns=np.array)
def block_rand(shape):
    return np.asfortranarray(np.random.random_sample(shape))


@task(block=INOUT)
def block_setitem(block, idx, value):
    block[idx] = value


@task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def block_tensordot(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@task(block=IN, returns=np.ndarray)
def block_copy(block):
    return np.copy(block)


@task(block=IN, returns=np.ndarray)
def block_pass(block):
    return block


@task(block=IN)
def block_getitem(block: np.ndarray, idx: tuple):
    return block[idx]


@task(returns=np.ndarray)
def block_kron(a: np.ndarray, b: np.ndarray):
    return np.kron(a, b)


@task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def block_gemm(a, b, transpose_a, transpose_b):
    return np.sum(__gemm(ai, bi, transpose_a, transpose_b) for ai, bi in zip(a, b))


# TODO do we need to do .conj() if complex number?
def __gemm(a, b, transpose_a, transpose_b):
    if transpose_a and transpose_b:
        return a.T @ b.T
    elif transpose_a and not transpose_b:
        return a.T @ b
    elif not transpose_a and transpose_b:
        return a @ b.T
    else:
        return a @ b


@task(returns=np.ndarray)
def block_identity(block_shape, n, i, j, dtype):
    block = np.zeros(block_shape, dtype)

    diag = np.intersect1d(*[np.arange(idx * bs, min(n, (idx+1)*bs))
                            for idx, bs in zip([i, j], block_shape)])

    i_ones, j_ones = [diag - idx * bs for idx, bs in zip([i, j], block_shape)]

    block[i_ones, j_ones] = 1


@task(U={Type: COLLECTION_INOUT, Depth: 2}, V={Type: COLLECTION_INOUT, Depth: 2})
def svdmatrix_async_blocked(U, V, eps: float):
    checks = [True]
    n = len(U)
    while any(compss_wait_on(checks)):
        checks = []

        for i, j in filter(lambda x: x[0] < x[1], itertools.product(range(n), range(n))):
            rot, check = _svd_compute_rotation(U[:][i], U[:][j], eps)
            _svd_rotate(U[:][i], U[:][j], rot)
            _svd_rotate(V[:][i], V[:][j], rot)

            checks.append(check)


# TODO COLLECTION_INOUT in order to preserve order?
@task(coli_blocks=COLLECTION_IN, colj_blocks=COLLECTION_IN, returns=2)
def _svd_compute_rotation(coli_blocks, colj_blocks, eps):
    coli = np.block(coli_blocks)
    colj = np.block(colj_blocks)

    bii = coli.T @ coli
    bjj = colj.T @ colj
    bij = coli.T @ colj

    if eps * np.sqrt(bii @ bjj) > np.linalg.norm(bij):
        # TODO bij.T.conj() for complex matrixes ?
        b = np.block([[bii, bij], [bij.T, bjj]])
        J, _, _ = np.linalg.svd(b)
        return J, True
    else:
        return None, False


# TODO try with Direction: COMMUTATIVE. not proven but theorically works.
@task(coli_blocks=COLLECTION_INOUT, colj_blocks=COLLECTION_INOUT)
def _svd_rotate(coli_blocks, colj_blocks, J):
    if J is None:
        return

    coli = np.block(coli_blocks)
    colj = np.block(colj_blocks)

    # TODO why all this?
    n = coli.shape[1]
    coli_k = coli @ J[:n, :n] + colj @ J[n:, :n]
    colj_k = coli @ J[:n, n:] + colj @ J[n:, n:]

    block_size = coli_blocks[0][0].shape[0]
    for i, _ in enumerate(coli_blocks):
        coli_blocks[i][0][:] = coli_k[i * block_size:(i + 1) * block_size][:]
        colj_blocks[i][0][:] = colj_k[i * block_size:(i + 1) * block_size][:]
