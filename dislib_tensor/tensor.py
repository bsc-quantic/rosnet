from typing import List, Tuple
import dislib
import numpy as np
from itertools import product, tee, accumulate, chain
from dislib.data.array import Array
from dislib_tensor import kernel
from dislib_tensor.utils import prod


class Tensor(object):
    """ A distributed n-rank tensor divided in blocks.

    Uses a dislib.Array as a backend.

    Parameters
    ----------


    Attributes
    ----------
    sign: array-like of str TODO
    shape: array-like of int
    I: array-like of int (shape of local indexes)
    J: array-like of int (shape of distributed indexes)
    """

    def __init__(self, blocks, shape, block_rank):
        self._shape = shape
        self._I = shape[0:block_rank]
        self._J = shape[block_rank:]
        self._blocks = blocks

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
        blocks = [kernel._block_full(block_shape, 0, dtype)
                  for _ in range(n_blocks)]

        return Tensor(blocks, shape, block_rank)

    @staticmethod
    def rand(shape, block_rank):
        assert len(shape) >= block_rank

        block_shape = shape[block_rank:]
        n_blocks = prod(block_shape)
        blocks = [kernel._block_rand(block_shape) for _ in range(n_blocks)]

        return Tensor(blocks, shape, block_rank)

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
        return NotImplemented("")

    @property
    def block_size(self):
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

    def permute(self, a, b):
        """ Permute index `a` with `b` """
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("Invalid type for _permute_dist args: a=%s, b=%s" % str(
                type(a)) % str(type(b)))
        assert 0 <= a < self.rank
        assert 0 <= b < self.rank

        # identify permutation => do nothing
        if a == b:
            return

        a, b = min(a, b), max(a, b)

        # distributed case
        if a >= self.block_rank and b >= self.block_rank:
            self._permute_dist(a, b)
        # local case
        elif a < self.block_rank and b < self.block_rank:
            self._permute_local(a, b)
        # hybrid case
        else:
            self._permute_hybrid(a, b)

        self._permute_shape(a, b)

    # TODO work when dim(a) != dim(b). should change the dislib.Array structure?
    def _permute_dist(self, a: int, b: int):
        """ Permute index `a` with `b` when both indexes are distributed """
        assert self.block_rank <= a < len(self.rank)
        assert self.block_rank <= b < len(self.rank)

        if self.shape[a] != self.shape[b]:
            raise NotImplementedError(
                "_permute_dist not implemented when dim(a) != dim(b)")

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

    def _permute_local(self, a, b):
        """ Permute index `a` with `b` when both indexes are local """
        assert 0 <= a < self.block_rank
        assert 0 <= b < self.block_rank

        # distribute permutation of blocks
        for block in self._blocks:
            kernel._block_permute(block, self.I, a, b)

    def _permute_hybrid(self, a, b):
        """ Permute index `a` with `b` when `a` is local and `b` is distributed """
        strides_J = self._stride_dist()

        print("[DEBUG] _permute_hybrid:")
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

    def _permute_shape(self, a, b):
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

        return product(*[range(i) for i in U])
