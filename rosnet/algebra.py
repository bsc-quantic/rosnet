from rosnet.tensor import Tensor, array
from rosnet.utils import isunique, space, prod, ispower2, ndarray_from_list
from rosnet import kernel, linalg
from copy import copy, deepcopy
from itertools import chain
import numpy as np
from pycompss.api.api import TaskGroup

OPTIONS = {
    'parallel-partials': False,
}


def transpose(t: Tensor, axes):
    # TODO is it ok to copy?
    return deepcopy(t).transpose(axes)


def tensordot(a: Tensor, b: Tensor, axes) -> Tensor:
    if isinstance(a, np.ndarray):
        a = array(a, a.shape)
    if isinstance(b, np.ndarray):
        b = array(b, b.shape)

    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("Invalid argument type: a=%s, b=%s" %
                        (type(a), type(b)))

    if not isinstance(axes, tuple) and not isinstance(axes, list):
        raise TypeError(
            "Invalid argument type: axes=%s and should be Tuple or List" % type(axes))

    if not isunique(axes[0]) or not isunique(axes[1]):
        raise ValueError("axes must not be repeated")

    if any(a.block_shape[axe_a] != b.block_shape[axe_b] for axe_a, axe_b in zip(*axes)):
        raise ValueError(
            "Cannnot contract Tensors with incompatible block-shape on contraction indexes")

    if a.shape == [] and b.shape == []:
        res = np.array(kernel.block_kron(a._blocks[()], b._blocks[()]))
        return Tensor(res, (), ())

    axes_a, axes_b = axes

    shape_a = list(map(lambda x: x[1], filter(
        lambda x: x[0] not in axes_a, enumerate(a.shape))))
    shape_b = list(map(lambda x: x[1], filter(
        lambda x: x[0] not in axes_b, enumerate(b.shape))))
    shape = shape_a + shape_b

    block_shape_a = list(map(lambda x: x[1], filter(
        lambda x: x[0] not in axes_a, enumerate(a.block_shape))))
    block_shape_b = list(map(lambda x: x[1], filter(
        lambda x: x[0] not in axes_b, enumerate(b.block_shape))))
    block_shape = block_shape_a + block_shape_b

    grid = [s // bs for s, bs in zip(shape, block_shape)]

    def coordrange(t: Tensor, coord: list, axes: tuple) -> list:
        c = copy(coord)
        axes = sorted(axes)
        for ax in axes:
            c.insert(ax, slice(t.grid[ax]))
        return tuple(c)

    # for each block in C
    tensorid = str(next(Tensor._newid))
    blocks = []
    with TaskGroup(tensorid, False):
        for coord in space(grid):
            # get all blocks in grid-a/grid-b with coord-a/coord-b + range(contraction indexes)
            coord_a, coord_b = coord[:len(shape_a)], coord[len(shape_a):]
            blocks_a = list(
                a._blocks[coordrange(a, list(coord_a), axes_a)].flat)
            blocks_b = list(
                b._blocks[coordrange(b, list(coord_b), axes_b)].flat)

            # block in C is equal to the sum of contractions of block pairs
            if OPTIONS['parallel-partials'] and shape != block_shape:
                # exploit inner parallelism of tensordot if enabled and tensor is sliced
                partials = [kernel.block_partialdot(
                    pa, pb, axes) for pa, pb in zip(blocks_a, blocks_b)]
                blocks.append(kernel.block_sum(partials))
            else:
                # do not exploit inner parallelism of tensordot if disabled or not sliced
                blocks.append(kernel.block_tensordot(blocks_a, blocks_b, axes))

    blocks = ndarray_from_list(blocks, grid)
    return Tensor(blocks, shape, block_shape, True, tensorid)


def kron(a: Tensor, b: Tensor) -> Tensor:
    shape = list(a.shape) + list(b.shape)
    block_shape = list(a.block_shape) + list(b.block_shape)
    grid = list(a.grid) + list(b.grid)
    blocks = np.empty(grid, dtype='object', order='F')
    n = a.rank

    with np.nditer(blocks, flags=['multi_index'], op_flags=['writeonly']) as it:
        for block in it:
            index = list(it.multi_index)
            block[...] = kernel.block_kron(
                a._blocks[index[:n]], b._blocks[index[n:]])

    return Tensor(blocks, shape, block_shape)


def schmidt(a: Tensor, axes_v, chi=None, eps=1e-9, copy=False) -> (Tensor, Tensor):
    """ Decomposes tensor `a` into two tensors `u` and `v` using the Schmidt decomposition.

    `a`: `Tensor` to decompose.

    `axes_v`: `tuple`-like. List of indexes kept by `v`.

    `chi`: `int`. Maximum rank of the decomposition. Currently unused.

    `eps`: `float`. Epsilon.
    """
    if not isinstance(axes_v, list) and not isinstance(axes_v, tuple):
        raise TypeError(
            f'axes_v must be of type "tuple" or "list"; {type(axes_v)} instead')
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

    shape_a = a.shape
    shape_u = [a.shape[i] for i in axes_u] + [k]
    shape_v = [a.shape[i] for i in axes_v] + [k]
    bshape_a = a.block_shape
    bshape_u = [a.block_shape[i] for i in axes_u]
    bshape_v = [a.block_shape[i] for i in axes_v]

    # permute tensor
    permutation = tuple(chain(axes_u, axes_v))
    a.transpose(permutation)

    # reshape to matrix
    a.reshape((m, n), (mb, nb))

    # perform SVD
    U, V = linalg.svd(a, eps)

    # reshape A, U, V to tensors
    bshape_u += [U.block_shape[1]]
    bshape_v += [V.block_shape[1]]
    a.reshape(shape_a, bshape_a)
    U.reshape(shape_u, bshape_u)
    V.reshape(shape_v, bshape_v)

    return (U, V)
