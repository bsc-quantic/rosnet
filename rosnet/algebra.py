from rosnet.tensor import Tensor
from rosnet.utils import isunique, space
from rosnet import kernel
from copy import copy, deepcopy
import numpy as np
from pycompss.api.api import TaskGroup


def transpose(t: Tensor, axes):
    # TODO is it ok to copy?
    return deepcopy(t).transpose(axes)


def tensordot(a: Tensor, b: Tensor, axes) -> Tensor:
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

            # block in C is equal to the sum of contractions of blocks
            # TODO are blocks_a, blocks_b in correct pair order?
            blocks.append(kernel.block_tensordot(blocks_a, blocks_b, axes))

    # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
    if isinstance(blocks[0], np.ndarray):
        bs = np.empty_like(range(len(blocks)), dtype=np.ndarray)
        for i in range(len(blocks)):
            bs[i] = blocks[i]
        blocks = bs.reshape(grid)
    else:
        blocks = np.array(blocks).reshape(grid)

    return Tensor(blocks, shape, block_shape, True, tensorid)
