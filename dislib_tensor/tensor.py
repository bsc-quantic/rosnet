from typing import List, Tuple
from pycompss.api.api import compss_delete_object, compss_barrier_group, compss_wait_on, TaskGroup
import numpy as np
from itertools import product, accumulate, count
from dislib.data.array import Array
from dislib_tensor import kernel
from dislib_tensor.utils import prod, space


class Tensor(object):
    """ A distributed n-rank tensor divided in blocks.

    Uses a dislib.Array as a backend.

    Parameters
    ----------


    Attributes
    ----------
    _shape: array-like of int
    _I: array-like of int (shape of local indexes)
    _J: array-like of int (shape of distributed indexes)
    _blocks: list of pyCOMPSs Futures
    _delete: bool
    """

    __newid = count().next

    def __init__(self, blocks, shape, block_rank, delete=True, tensorid=None):
        self._shape = shape
        self._I = shape[0:block_rank]
        self._J = shape[block_rank:]
        self._blocks = blocks
        self._delete = delete
        self._tensorid = tensorid if tensorid != None else Tensor.__newid()

    def __del__(self):
        if self._delete:
            [compss_delete_object(block) for block in self._blocks]

    def __str__(self):
        return "Tensor(blocks=(...), shape=%r, block_rank=%r)" % (self.shape, self.block_rank)

    def __repr__(self):
        return "Tensor(blocks=(...), shape=%r, block_rank=%r)" % (self.shape, self.block_rank)

    def __getitem__(self, arg):
        if isinstance(arg, list) and isinstance(arg[0], int):
            if len(arg) != self.rank:
                raise IndexError("Invalid indexing information: arg=%s" % arg)

            local_id, dist_id = arg[:self.block_rank], arg[self.block_rank:]
            strides_J = self._stride_dist()
            dist_id = sum(c * s for c, s in zip(dist_id, strides_J))

            block = compss_wait_on(self._blocks[dist_id], write=False)
            return block[local_id]

        raise IndexError("Invalid indexing information: %s" % arg)

    def __setitem__(self, key, value):
        if isinstance(key, list) and isinstance(key[0], int):
            if len(key) != self.rank:
                raise IndexError("Invalid indexing information: key=%s" % key)

            local_id, dist_id = key[:self.block_rank], key[self.block_rank:]
            strides_J = self._stride_dist()
            dist_id = sum(c * s for c, s in zip(dist_id, strides_J))

            kernel._block_set_value(self._blocks[dist_id], local_id, value)

        raise IndexError("Invalid indexing information: %s" % key)

    @staticmethod
    def zeros(shape, block_rank, dtype=None):
        return Tensor.full(0, shape, block_rank, dtype)

    @staticmethod
    def ones(shape, block_rank, dtype=None):
        return Tensor.full(1, shape, block_rank, dtype)

    @staticmethod
    def full(value, shape, block_rank, dtype=None):
        assert len(shape) >= block_rank

        block_shape = shape[block_rank:]
        n_blocks = prod(block_shape)

        tensorid = Tensor.__newid()
        with TaskGroup(tensorid, False):
            blocks = [kernel._block_full(block_shape, 0, dtype)
                      for _ in range(n_blocks)]

        return Tensor(blocks, shape, block_rank, True, tensorid)

    @staticmethod
    def rand(shape, block_rank):
        assert len(shape) >= block_rank

        block_shape = shape[block_rank:]
        n_blocks = prod(block_shape)

        tensorid = Tensor.__newid()
        with TaskGroup(tensorid, False):
            blocks = [kernel._block_rand(block_shape) for _ in range(n_blocks)]

        return Tensor(blocks, shape, block_rank, True, tensorid)

    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return len(self.shape)

    @property
    def block_rank(self):
        return len(self.I)

    @block_rank.setter
    def _(self):
        return NotImplementedError("")

    @property
    def block_volume(self):
        return prod(self.I)

    @property
    def block_num(self):
        return prod(self.J)

    @property
    def I(self):
        return self._I

    @property
    def J(self):
        return self._J

    def volume(self):
        return prod(self.shape)

    def sync(self):
        compss_barrier_group(self._tensorid)

    # TODO impl with args (self, axes)
    def transpose(self, a, b):
        """ Permute index `a` with `b` """
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("Invalid type for _transpose_dist args: a=%s, b=%s" % str(
                type(a)) % str(type(b)))
        assert 0 <= a < self.rank
        assert 0 <= b < self.rank

        # identity transposition => do nothing
        if a == b:
            return

        a, b = min(a, b), max(a, b)

        with TaskGroup(self._tensorid, False):
            # distributed case
            if a >= self.block_rank:  # and b >= self.block_rank:
                self._transpose_dist(a, b)
            # local case
            elif b < self.block_rank:  # a < self.block_rank and
                self._transpose_local(a, b)
            # hybrid case
            else:
                self._transpose_hybrid(a, b)

        self._transpose_shape(a, b)

    def matrix(self, t: int):
        """ Returns a `dislib.data.array.Array` representing the matrix view of the tensor """
        # Block is 1D if t >= self.block_rank
        # Block List is 1D if t <= self.block_rank
        assert 0 <= t < self.rank

        shape = (self.shape[:t], self.shape[t:])
        if t < self.block_rank:
            # Blocks are 2D chunks and Block List is 1D
            reg_shape = (prod(self.I[:t]), prod(self.I[t:]))
            blocks = [self._blocks]
        elif t == self.block_rank:
            # Blocks are 1D chunks and Block List is 1D
            reg_shape = (prod(self.I), 1)
            blocks = [self._blocks]
        else:
            # Blocks are 1D chunks and Block List is 2D
            reg_shape = (prod(self.I), 1)
            blocks = np.array(self._blocks).reshape(
                (prod(self.J[:t - self.block_rank], prod(self.J[t-self.block_rank:])))).tolist()

        n_blocks = (shape[0] // reg_shape[0], shape[1] // reg_shape[1])

        return Array(blocks, reg_shape, reg_shape, shape, False, False)

    # TODO work when dim(a) != dim(b). should change the dislib.Array structure?
    def _transpose_dist(self, a: int, b: int):
        """ Permute index `a` with `b` when both indexes are distributed """
        assert self.block_rank <= a < len(self.rank)
        assert self.block_rank <= b < len(self.rank)

        if self.shape[a] != self.shape[b]:
            raise NotImplementedError(
                "_transpose_dist not implemented when dim(a) != dim(b)")

        a = a - self.block_rank
        b = b - self.block_rank

        stride = self._stride_dist()
        def linear(c): return sum([i * j for i, j in zip(c, stride)])

        # space of blocks
        def u_space(x, y):
            for i in self._u_space([x, y]):
                i = list(i)
                if a < b:
                    i.insert(a, x)
                    i.insert(b, y)
                else:
                    i.insert(b, y)
                    i.insert(a, x)
                yield i

        # TODO work with dim(a) != dim(b)
        def ab_space(a, b):
            major, minor = (0, 1) if self.J[a] >= self.J[b] else (1, 0)
            return filter(lambda x: x[0] != x[1] and x[minor] < x[major], product(
                range(self.J[a]), range(self.J[b])))

        # travel space and swap blocks
        for x, y in ab_space(a, b):
            for ca in u_space(x, y):
                # get coord of 2nd block
                cb = ca
                cb[a], cb[b] = ca[b], ca[a]

                # ids of swapping blocks
                ba = linear(ca)
                bb = linear(cb)

                # swap blocks
                self._blocks[ba], self._blocks[bb] = self._blocks[bb], self._blocks[ba]

    def _transpose_local(self, a, b):
        """ Permute index `a` with `b` when both indexes are local """
        assert 0 <= a < self.block_rank
        assert 0 <= b < self.block_rank

        # distribute permutation of blocks
        for block in self._blocks:
            kernel._block_transpose(block, self.I, a, b)

    def _transpose_hybrid(self, a, b):
        """ Permute index `a` with `b` when `a` is local and `b` is distributed """
        strides_J = self._stride_dist()

        print("[DEBUG] _transpose_hybrid:")
        print("\tlocal -> a = %s" % str(a))
        print("\tdist -> b = %s" % str(b))
        print("\tstrides_J = %s" % str(list(strides_J)))
        print("\t%s" % str(type(strides_J)))

        b = b - self.block_rank

        # traverse U space
        for u in self._u_space(b):
            u = list(u)

            # select all matching blocks
            def copy_insert(x, y):
                r = u.copy()
                r.insert(x, y)
                return r
            blocks_coord = [copy_insert(b, b_id) for b_id in range(self.J[b])]

            blocks_linear = [sum(coord * stride for coord, stride in zip(
                block_coord, strides_J)) for block_coord in blocks_coord]

            blocks = [self._blocks[i] for i in blocks_linear]
            kernel._block_merge_split(blocks, self.I, a)

    def _transpose_shape(self, a, b):
        self._shape[a], self._shape[b] = self._shape[b], self._shape[a]
        block_rank = self.block_rank
        self._I = self._shape[0:block_rank]
        self._J = self._shape[block_rank:]

    def _stride_local(self):
        return list(accumulate([1] + self.I[:-1], lambda x, y: x * y))

    def _stride_dist(self):
        return list(accumulate([1] + self.J[:-1], lambda x, y: x * y))

    def _u_space(self, b):
        """ Generates the space of U where U = J - {b}
        This space is used to traverse the blocks with the same `b` value or to gather all the blocks in the `b` index within the same U-space coordinate.
        """
        U = self.J.copy()
        if isinstance(b, List):
            assert all(self.block_rank <= i < self.rank for i in b)
            b.sort(reverse=True)
            for i in b:
                del U[i]
        elif isinstance(b, int):
            assert self.block_rank <= b < self.rank
            del U[b]
        else:
            raise TypeError("Invalid type for _u_space: %s" % str(type(b)))

        return space(U)


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

    # TODO do transpositions
    raise NotImplementedError("")

    # TODO transform to dislib.Array and perform matrix multiplication
    matrix_a = a.matrix(t)
    matrix_b = b.matrix()  # TODO

    #  TODO transform back to Tensor
    raise NotImplementedError("")
