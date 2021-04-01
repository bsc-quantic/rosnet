from pycompss.api.api import compss_delete_object, compss_barrier_group, compss_wait_on, TaskGroup
import numpy as np
from itertools import product, count
from dislib.data.array import Array
from dislib_tensor import kernel
from dislib_tensor.utils import prod


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

    __newid = count().next

    def __init__(self, blocks, shape, block_shape, delete=True, tensorid=None):
        if not isinstance(blocks, np.ndarray):
            raise TypeError(
                "Invalid argument type: blocks=%s and must be numpy.ndarray" % type(blocks))
        if not isinstance(shape, tuple):
            raise TypeError(
                "Invalid argument type: shape=%s and must be tuple" % type(shape))
        if not isinstance(block_shape, tuple):
            raise TypeError(
                "Invalid argument type: block_shape=%s and must be tuple" % type(block_shape))
        if not isinstance(delete, bool):
            raise TypeError(
                "Invalid argument type: delete=%s and must be bool" % type(delete))
        if not isinstance(tensorid, int):
            raise TypeError(
                "Invalid argument type: tensorid=%s and must be int" % type(tensorid))

        self._shape = shape
        self._block_shape = block_shape
        self._blocks = blocks
        self._delete = delete
        self._tensorid = tensorid if tensorid != None else Tensor.__newid()

    def __del__(self):
        if self._delete:
            for block in np.nditer(self._blocks):
                compss_delete_object(block)

    def __str__(self):
        return "Tensor(blocks=(...), shape=%r, block_shape=%r)" % (self.shape, self.block_shape)

    def __repr__(self):
        return "Tensor(blocks=(...), shape=%r, block_shape=%r)" % (self.shape, self.block_shape)

    def __getitem__(self, arg):
        if isinstance(arg, list) and isinstance(arg[0], int):
            if len(arg) != self.rank:
                raise IndexError("Invalid indexing information: arg=%s" % arg)

            grid_id = [i // s for i, s in zip(arg, self.block_shape)]
            block_id = [i % s for i, s in zip(arg, self.block_shape)]
            block = compss_wait_on(self._blocks[grid_id], write=False)
            return block[block_id]

        raise IndexError("Invalid indexing information: %s" % arg)

    def __setitem__(self, key, value):
        if isinstance(key, list) and isinstance(key[0], int):
            if len(key) != self.rank:
                raise IndexError("Invalid indexing information: key=%s" % key)

            grid_id = [i // s for i, s in zip(key, self.block_shape)]
            block_id = [i % s for i, s in zip(key, self.block_shape)]
            kernel._block_set_value(self._blocks[grid_id], block_id, value)

        raise IndexError("Invalid indexing information: %s" % key)

    @staticmethod
    def zeros(shape, block_shape, dtype=None):
        return Tensor.full(0, shape, block_shape, dtype)

    @staticmethod
    def ones(shape, block_shape, dtype=None):
        return Tensor.full(1, shape, block_shape, dtype)

    @staticmethod
    def full(value, shape, block_shape, dtype=None):
        grid = [s // bs for s, bs in zip(shape, block_shape)]

        tensorid = Tensor.__newid()
        with TaskGroup(tensorid, False):
            blocks = [kernel._block_full(block_shape, value, dtype)
                      for _ in range(prod(grid))]

        blocks = np.array(blocks).reshape(shape)
        return Tensor(blocks, shape, block_shape, True, tensorid)

    @staticmethod
    def rand(shape, block_shape):
        grid = [s // bs for s, bs in zip(shape, block_shape)]

        tensorid = Tensor.__newid()
        with TaskGroup(tensorid, False):
            blocks = [kernel._block_rand(block_shape)
                      for _ in range(prod(grid))]

        blocks = np.array(blocks).reshape(shape)
        return Tensor(blocks, shape, block_shape, True, tensorid)

    @property
    def shape(self):
        return self._shape

    @property
    def block_shape(self):
        return self._block_shape

    @block_shape.setter
    def _(self):
        return NotImplementedError("")

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
        for block in np.nditer(self._blocks):
            kernel._block_transpose(block, axes)

        for a, b in zip(range(self.rank), axes):
            self._shape[a], self._shape[b] = self._shape[b], self._shape[a]


def transpose(a: Tensor, axes) -> Tensor:
    if not isinstance(a, Tensor):
        raise TypeError("Invalid argument type: a=%s" % type(a))

    if not isinstance(axes, tuple) or not isinstance(axes, list):
        raise TypeError("Invalid argument type: axes=%s" % type(axes))

    raise NotImplementedError("")


def tensordot(a: Tensor, b: Tensor, axes) -> Tensor:
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("Invalid argument type: a=%s, b= %s" %
                        (type(a), type(b)))

    axes_a, axes_b = axes
    if len(axes_a) != axes_b:
        raise ValueError("Invalid value type: axes=%s" % axes)
    t = len(axes_a)

    # TODO
    raise NotImplementedError("")
