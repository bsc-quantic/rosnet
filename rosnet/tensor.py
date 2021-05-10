from pycompss.api.api import compss_delete_object, compss_barrier_group, compss_wait_on, TaskGroup
import numpy as np
from itertools import product, count
from rosnet import kernel
from rosnet.utils import prod, isunique, space


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

    _newid = count()

    def __init__(self, blocks, shape, block_shape=None, delete=True, tensorid=None):
        block_shape = shape if block_shape == None else block_shape

        if not isinstance(blocks, np.ndarray):
            raise TypeError(
                "Invalid argument type: blocks=%s and must be numpy.ndarray" % type(blocks))
        if not isinstance(shape, list) and not isinstance(shape, tuple):
            raise TypeError(
                "Invalid argument type: shape=%s and must be list" % type(shape))
        shape = list(shape)
        if not isinstance(block_shape, list) and not isinstance(block_shape, tuple):
            raise TypeError(
                "Invalid argument type: block_shape=%s and must be list" % type(block_shape))
        block_shape = list(block_shape)
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
            next(Tensor._newid))

    def __del__(self):
        if self._delete:
            for block in self._blocks.flat:
                compss_delete_object(block)

    def __deepcopy__(self):
        blocks = np.empty_like(self._blocks)

        for i, block in enumerate(self._blocks.flat):
            blocks[i] = kernel.block_copy(block)

        return Tensor(blocks, self.shape, self.block_shape)

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

        tensorid = str(next(Tensor._newid))
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

        tensorid = str(next(Tensor._newid))
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

        tensorid = str(next(Tensor._newid))
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

    @property
    def volume(self):
        return prod(self.shape)

    def sync(self):
        compss_barrier_group(self._tensorid)

    def transpose(self, axes):
        if not isunique(axes):
            raise ValueError("axes must be a unique list: %s" % axes)

        # transpose blocks
        with TaskGroup(self._tensorid):
            for i in range(self._blocks.size):
                self._blocks.flat[i] = kernel.block_transpose(
                    self._blocks.flat[i], axes)

        # tranpose grid
        self._blocks = np.transpose(self._blocks, axes)

        # transpose shapes
        self._shape = [self._shape[i] for i in axes]
        self._block_shape = [self._block_shape[i] for i in axes]

    def reshape(self, shape, block_shape):
        if prod(shape) != self.volume:
            raise ValueError("new shape must not change volume")

        grid = tuple(s // bs for s, bs in zip(shape, block_shape))
        with TaskGroup(self._tensorid):
            for i in range(self._blocks.size):
                self._blocks.flat[i] = kernel.block_reshape(
                    self._blocks.flat[i], block_shape)

        self._blocks = self._blocks.reshape(grid, order='F')
        self._shape = shape
        self._block_shape = block_shape

    def rechunk(self, shape):
        """ Redefine the shape of the tensor block
        """
        assert len(shape) == self.rank
        assert all(i < j for i, j in zip(shape, self.shape))
        assert all(i % j == 0 for i, j in zip(self.shape, shape))

        for axis, (block_dim, new_dim) in enumerate(zip(self.block_shape, shape)):
            if block_dim == new_dim:
                continue

            elif block_dim % new_dim == 0:
                # split blocks
                self._rechunk_split(block_dim / new_dim, axis)

            elif new_dim % block_dim == 0:
                # merge blocks
                self._rechunk_merge(new_dim / block_dim, axis)

            else:
                raise NotImplementedError(
                    "Rechunking by a non-multiple or non-divisible factor is not supported")

        self._block_shape = shape

    def _rechunk_split(self, n: int, axis: int):
        new_shape = list(self.grid)
        new_shape[axis] = new_shape[axis] * n
        grid = np.empty(new_shape, dtype=object, order='F')

        with np.nditer(self._blocks, flags=['multi_index', 'refs_ok']) as it:
            for block in it:
                collection = kernel.block_split(block, n, axis)
                index = list(it.multi_index)
                index[axis] = slice(index[axis], index[axis] + n)
                grid[index] = collection

        self._blocks = grid

    def _rechunk_merge(self, n: int, axis: int):
        new_shape = list(self.grid)
        new_shape[axis] = new_shape[axis] / n
        grid = np.empty(new_shape, dtype=object, order='F')

        with np.nditer(grid, flags=['multi_index'], op_flags=['writeonly']) as it:
            for block in it:
                index = list(it.multi_index)
                index[axis] = slice(index[axis], index[axis] + n)
                collection = self._blocks[index]
                # TODO block or block[...]. check numpy.nditer doc
                block = kernel.block_merge(collection, axis)

        self._blocks = grid
