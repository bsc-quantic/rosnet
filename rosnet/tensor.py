from pycompss.api.api import compss_delete_object, compss_barrier_group, compss_wait_on, TaskGroup
import numpy as np
from itertools import product, count, repeat
from rosnet import kernel
from rosnet.utils import prod, isunique, space
from copy import copy


class Tensor(object):
    """ A distributed n-rank tensor divided in blocks.

    Parameters
    ----------
    blocks: ndarray
    shape: tuple
        Shape of the whole tensor.
    block_shape: tuple
        Shape of a tensor block. NOTE: The rank of a tensor block must equal the rank of the whole tensor so `len(block_shape) == len(shape)`.
    delete: bool, optional (default=True)
        Call `compss_delete_object` on the blocks when the garbage collector deletes this object.
    tensorid: int, optional (default=None).

    Attributes
    ----------
    _shape: array-like of int
    _block_shape
    _blocks: list of pyCOMPSs Futures
    _delete: bool
    """

    __newid = count()

    def __init__(self, blocks, shape, block_shape=None, delete=True, tensorid=None):
        block_shape = shape if block_shape == None else block_shape

        if not isinstance(blocks, np.ndarray):
            raise TypeError(
                "Invalid argument type: blocks=%s and must be numpy.ndarray" % type(blocks))
        if not isinstance(shape, list):
            raise TypeError(
                "Invalid argument type: shape=%s and must be list" % type(shape))
        if not isinstance(block_shape, list):
            raise TypeError(
                "Invalid argument type: block_shape=%s and must be list" % type(block_shape))
        if not isinstance(delete, bool):
            raise TypeError(
                "Invalid argument type: delete=%s and must be bool" % type(delete))
        if not isinstance(tensorid, str) and tensorid != None:
            raise TypeError(
                "Invalid argument type: tensorid=%s and must be str" % type(tensorid))

        if not all(s % bs == 0 for s, bs in zip(shape, block_shape)):
            raise ValueError("Invalid argument value: shape=%s must be divisible by block_shape=%s" % (
                shape, block_shape))

        grid = tuple(s // bs for s, bs in zip(shape, block_shape))
        if blocks.shape != grid:
            raise ValueError("Invalid argument value: blocks.shape=%s and grid=%s should be equal" % (
                blocks.shape, grid))

        self._shape = shape
        self._block_shape = block_shape
        self._blocks = blocks
        self._delete = delete
        self._tensorid = tensorid if tensorid != None else str(
            next(Tensor.__newid))

    def __del__(self):
        if self._delete:
            for block in self._blocks.flat:
                compss_delete_object(block)

    def __str__(self):
        return "Tensor(blocks=(...), shape=%r, grid=%r, block_shape=%r)" % (self.shape, self.grid, self.block_shape)

    def __repr__(self):
        return "Tensor(blocks=(...), shape=%r, grid=%r, block_shape=%r)" % (self.shape, self.grid, self.block_shape)

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            arg = list(arg)
        if isinstance(arg, list) and isinstance(arg[0], int):
            if len(arg) != self.rank:
                raise IndexError("Invalid indexing information: arg=%s" % arg)

            grid_id = tuple(i // s for i, s in zip(arg, self.block_shape))
            block_id = tuple(i % s for i, s in zip(arg, self.block_shape))
            return compss_wait_on(kernel.block_getitem(self._blocks[grid_id], block_id))

        raise IndexError("Invalid indexing information: %s" % arg)

    def __setitem__(self, key, value):
        if isinstance(key, list) and isinstance(key[0], int):
            if len(key) != self.rank:
                raise IndexError("Invalid indexing information: key=%s" % key)

            grid_id = [i // s for i, s in zip(key, self.block_shape)]
            block_id = [i % s for i, s in zip(key, self.block_shape)]
            kernel.block_setitem(self._blocks[grid_id], block_id, value)

        raise IndexError("Invalid indexing information: %s" % key)

    @staticmethod
    def array(arr: np.ndarray, block_shape):
        shape = arr.shape
        grid = [s // bs for s, bs in zip(shape, block_shape)]

        tensorid = str(next(Tensor.__newid))
        blocks = []
        with TaskGroup(tensorid, False):
            for bidx in space(grid):
                idx_begin = [x*y for x, y in zip(block_shape, bidx)]
                idx_end = [x+y for x, y in zip(block_shape, idx_begin)]
                idx = tuple(slice(b, e) for b, e in zip(idx_begin, idx_end))
                block = arr[idx]
                blocks.append(kernel.block_pass(block))

        # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
        if isinstance(blocks[0], np.ndarray):
            bs = np.empty_like(range(len(blocks)), dtype=np.ndarray)
            for i in range(len(blocks)):
                bs[i] = blocks[i]
            blocks = bs.reshape(grid)
        else:
            blocks = np.array(blocks).reshape(grid)
        return Tensor(blocks, list(shape), block_shape, True, tensorid)

    @staticmethod
    def zeros(shape, block_shape, dtype=None):
        return Tensor.full(0, shape, block_shape, dtype)

    @staticmethod
    def ones(shape, block_shape, dtype=None):
        return Tensor.full(1, shape, block_shape, dtype)

    @staticmethod
    def full(value, shape, block_shape, dtype=None):
        grid = tuple(s // bs for s, bs in zip(shape, block_shape))

        tensorid = str(next(Tensor.__newid))
        with TaskGroup(tensorid, False):
            blocks = [kernel.block_full(block_shape, value, dtype)
                      for _ in range(prod(grid))]

        # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
        if isinstance(blocks[0], np.ndarray):
            bs = np.empty_like(range(len(blocks)), dtype=np.ndarray)
            for i in range(len(blocks)):
                bs[i] = blocks[i]
            blocks = bs.reshape(grid)
        else:
            blocks = np.array(blocks).reshape(grid)
        return Tensor(blocks, shape, block_shape, True, tensorid)

    @staticmethod
    def rand(shape, block_shape):
        grid = tuple(s // bs for s, bs in zip(shape, block_shape))

        tensorid = str(next(Tensor.__newid))
        with TaskGroup(tensorid, False):
            blocks = [kernel.block_rand(block_shape)
                      for _ in range(prod(grid))]

        # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
        if isinstance(blocks[0], np.ndarray):
            bs = np.empty_like(range(len(blocks)), dtype=np.ndarray)
            for i in range(len(blocks)):
                bs[i] = blocks[i]
            blocks = bs.reshape(grid)
        else:
            blocks = np.array(blocks).reshape(grid)
        return Tensor(blocks, shape, block_shape, True, tensorid)

    @property
    def shape(self):
        return self._shape

    @property
    def block_shape(self):
        return self._block_shape

    @block_shape.setter
    def _(self, value):
        self.rechunk(value)

    @property
    def rank(self):
        return len(self.shape)

    @property
    def block_volume(self):
        return prod(self.block_shape)

    @property
    def grid(self):
        return self._blocks.shape

    @property
    def block_num(self):
        raise prod(self.grid)

    def volume(self):
        return prod(self.shape)

    def sync(self):
        compss_barrier_group(self._tensorid)

    def transpose(self, axes):
        if not isunique(axes):
            raise ValueError("axes must be a unique list: %s" % axes)

        # transpose blocks
        for i in range(self._blocks.size):
            self._blocks.flat[i] = kernel.block_transpose(
                self._blocks.flat[i], axes)

        # tranpose grid
        self._blocks = np.transpose(self._blocks, axes)

        # transpose shapes
        self._shape = [self._shape[i] for i in axes]
        self._block_shape = [self._block_shape[i] for i in axes]

    def rechunk(self, shape):
        raise NotImplementedError("coming soon!")

    def _getblocks(self, idx: list) -> list:
        if len(idx) != self.rank:
            raise TypeError(
                "Invalid indexing: idx must have as many elements as the rank of the tensor")

        # find range elements
        axes = list(map(lambda x: x[1], filter(
            lambda x: isinstance(x[1], slice) or isinstance(x[1], range), enumerate(idx))))

        # if no ranges, directly return the block
        if axes == []:
            return [self.grid[idx]]

        coords = [i if isinstance(i, slice) or isinstance(i, range) else repeat(
            i, times=1) for i in idx]
        return [self._blocks[coord] for coord in product(*coords)]


def transpose(a: Tensor, axes) -> Tensor:
    if not isinstance(a, Tensor):
        raise TypeError("Invalid argument type: a=%s" % type(a))

    if not isinstance(axes, tuple) or not isinstance(axes, list):
        raise TypeError("Invalid argument type: axes=%s" % type(axes))

    a.transpose(axes)
    return a


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

    axes_sorted = [sorted(ax, reverse=True) for ax in axes]
    shape_a = copy(a.shape)
    shape_b = copy(b.shape)
    for i, j in zip(*axes_sorted):
        del shape_a[i]
        del shape_b[j]
    shape = shape_a + shape_b

    block_shape_a = copy(a.block_shape)
    block_shape_b = copy(b.block_shape)
    for i, j in zip(*axes_sorted):
        del block_shape_a[i]
        del block_shape_b[j]
    block_shape = block_shape_a + block_shape_b

    grid = [s // bs for s, bs in zip(shape, block_shape)]
    blocks = np.ndarray(grid, dtype=object)

    def coordrange(t: Tensor, coord: list, axes: tuple) -> list:
        c = copy(coord)
        axes = sorted(axes)
        for ax in axes:
            c.insert(ax, range(t.grid[ax]))
        return c

    # for each block in C
    blocks = []
    for coord in space(grid):
        # get all blocks in grid-a/grid-b with coord-a/coord-b + range(contraction indexes)
        coord_a, coord_b = coord[:len(shape_a)], coord[len(shape_a):]
        axes_a, axes_b = sorted(axes[0]), sorted(axes[1])
        blocks_a = a._getblocks(coordrange(a, list(coord_a), axes_a))
        blocks_b = b._getblocks(coordrange(b, list(coord_b), axes_b))

        # block in C is equal to the sum of contractions of blocks
        blocks.append(kernel.block_tensordot(blocks_a, blocks_b, axes))

    # NOTE numpy reads 'blocks' recursively, so generate it manually when pycompss is deactivated
    if isinstance(blocks[0], np.ndarray):
        bs = np.empty_like(range(len(blocks)), dtype=np.ndarray)
        for i in range(len(blocks)):
            bs[i] = blocks[i]
        blocks = bs.reshape(grid)
    else:
        blocks = np.array(blocks).reshape(grid)

    return Tensor(blocks, shape, block_shape)
