from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.implement import implement
from pycompss.api.parameter import IN, COLLECTION_IN, COLLECTION_INOUT, Type, Depth
from pycompss.api.api import compss_wait_on
import numpy as np
import itertools
import scipy.linalg as la


@task(U={Type: COLLECTION_INOUT, Depth: 2}, Vh={Type: COLLECTION_INOUT, Depth: 2})
def svdmatrix_async(U, Vh):
    Ut = np.block(U)

    Ut, s, Vth = la.svd(Ut, full_matrices=False, overwrite_a=True)
    Ut = Ut * s
    Vth = Vth.transpose().conjugate()

    m, n = Ut.shape
    mb, nb = U[0][0].shape
    for i, j in itertools.product(range(m // mb), range(n // nb)):
        np.copyto(U[i][j], Ut[i*mb:(i+1)*mb, j*nb:(j+1)*nb], casting='no')
    for i, j in itertools.product(range(n // nb), range(n // nb)):
        np.copyto(Vh[i][j], Vth[i*nb:(i+1)*nb, j*nb:(j+1)*nb], casting='no')


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


# TODO try with Direction: COMMUTATIVE. not proven but theoretically works.
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
        np.copyto(coli_blocks[i], coli_k[i * block_size:(i + 1) * block_size])
        np.copyto(colj_blocks[i], colj_k[i * block_size:(i + 1) * block_size])
