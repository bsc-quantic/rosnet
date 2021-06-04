import os
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import Type, Depth, IN, INOUT, COLLECTION_INOUT, COLLECTION_IN, COLLECTION_OUT
from pycompss.api.api import compss_wait_on
import itertools
import scipy.linalg as la


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


@task(block=IN, returns=np.ndarray)
def block_transpose(block: np.ndarray, permutator):
    return transpose(block, permutator)


@task(block=IN, returns=np.ndarray)
def block_reshape(block: np.ndarray, shape: tuple):
    return block.reshape(shape, order='F')


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


@task(ba=IN, bb=IN, returns=np.ndarray)
def block_partialdot(ba, bb, axes):
    return np.tensordot(ba, bb, axes)


@task(blocks={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def block_sum(blocks):
    return sum(blocks)


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

    return block


@task(U={Type: COLLECTION_INOUT, Depth: 2}, Vh={Type: COLLECTION_INOUT, Depth: 2})
def svdmatrix_async(U, Vh):
    Ut = np.block(U)

    Ut, s, Vth = la.svd(Ut, full_matrices=False, overwrite_a=True)
    Ut = Ut * s
    Vth = Vth.transpose().conjugate()

    m, n = Ut.shape
    mb, nb = U[0][0].shape
    for i, j in itertools.product(range(m // mb), range(n // nb)):
        U[i][j][:] = Ut[i*mb:(i+1)*mb, j*nb:(j+1)*nb][:]
    for i, j in itertools.product(range(n // nb), range(n // nb)):
        Vh[i][j][:] = Vth[i*nb:(i+1)*nb, j*nb:(j+1)*nb][:]


@task(U={Type: COLLECTION_INOUT, Depth: 2}, V={Type: COLLECTION_INOUT, Depth: 2})
def svdmatrix_async_nested(U, V, eps: float):
    checks = [True]
    n = len(U[0])
    while any(compss_wait_on(checks)):
        checks = []

        for i, j in filter(lambda x: x[0] < x[1], itertools.product(range(n), range(n))):
            Ui = [Urow[i] for Urow in U]
            Uj = [Urow[j] for Urow in U]
            Vi = [Vrow[i] for Vrow in V]
            Vj = [Vrow[j] for Vrow in V]
            rot, check = _svd_compute_rotation(Ui, Uj, eps)
            _svd_rotate(Ui, Uj, rot)
            _svd_rotate(Vi, Vj, rot)

            checks.append(check)


# TODO COLLECTION_INOUT in order to preserve order?
@task(coli_blocks=COLLECTION_IN, colj_blocks=COLLECTION_IN, returns=2)
def _svd_compute_rotation(coli_blocks, colj_blocks, eps):
    coli = np.vstack(coli_blocks)
    colj = np.vstack(colj_blocks)

    bii = coli.T @ coli
    bjj = colj.T @ colj
    bij = coli.T @ colj

    if np.linalg.norm(bij) >= eps * np.sqrt(np.sum(bii * bjj)):
        # TODO bij.T.conj() for complex matrixes ?
        b = np.block([[bii, bij], [bij.T, bjj]])
        J, _, _ = np.linalg.svd(b)
        return J, True
    else:
        return None, False


# TODO try with Direction: COMMUTATIVE. not proven but theorically works.
@task(coli_blocks=COLLECTION_INOUT, colj_blocks=COLLECTION_INOUT, J=IN)
def _svd_rotate(coli_blocks, colj_blocks, J):
    if J is None:
        return

    coli = np.vstack(coli_blocks)
    colj = np.vstack(colj_blocks)

    n = coli.shape[1]
    coli_k = coli @ J[:n, :n] + colj @ J[n:, :n]
    colj_k = coli @ J[:n, n:] + colj @ J[n:, n:]

    block_size = coli_blocks[0].shape[0]
    for i, _ in enumerate(coli_blocks):
        coli_blocks[i][:] = coli_k[i * block_size:(i + 1) * block_size][:]
        colj_blocks[i][:] = colj_k[i * block_size:(i + 1) * block_size][:]
