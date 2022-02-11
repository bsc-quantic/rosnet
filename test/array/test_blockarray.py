import pytest
from typing import Tuple
from math import prod
import numpy as np
from rosnet import BlockArray


class TestConstructors:
    gridshape = (2, 2, 2)
    blockshape = (4, 8, 1)
    dtype = np.float64

    @property
    def shape(self) -> Tuple[int]:
        return tuple(g * b for g, b in zip(self.gridshape, self.blockshape))

    def test_nested_list_of_ndarray(self):
        data = np.empty(self.gridshape, dtype=object)
        it = np.nditer(data, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

        with it:
            for block in it:
                block[()] = np.empty(self.blockshape, dtype=self.dtype)

        arr = BlockArray(data.tolist())

        assert arr.grid == self.gridshape
        assert arr.blockshape == self.blockshape
        assert arr.dtype == self.dtype

    def test_list_of_ndarray(self):
        data = [np.zeros(self.blockshape, dtype=self.dtype)] * prod(self.gridshape)

        arr = BlockArray(data, grid=self.gridshape)

        assert arr.grid == self.gridshape
        assert arr.blockshape == self.blockshape
        assert arr.dtype == self.dtype

    def test_nested_ndarray(self):
        data = np.empty(self.gridshape, dtype=object)
        it = np.nditer(data, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

        with it:
            for block in it:
                block[()] = np.empty(self.blockshape, dtype=self.dtype)

        arr = BlockArray(data)

        assert arr.grid == self.gridshape
        assert arr.blockshape == self.blockshape
        assert arr.dtype == self.dtype

    def test_ndarray(self):
        data = np.zeros(self.blockshape, dtype=self.dtype)
        arr = BlockArray(data)

        assert arr.grid == tuple(1 for _ in self.gridshape)
        assert arr.blockshape == arr.shape == self.blockshape
        assert arr.dtype == self.dtype

    def test_array(self):
        blockshape = self.blockshape
        dtype = self.dtype

        class MockArray:
            def __array__(self):
                return np.zeros(blockshape, dtype=dtype)

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return NotImplemented

            def __array_function__(self, function, types, args, kwargs):
                return NotImplemented

            def __getitem__(self, key):
                return 0

            @property
            def shape(self) -> Tuple[int]:
                return blockshape

            @property
            def dtype(self) -> np.dtype:
                return dtype

        data = MockArray()
        arr = BlockArray(data)

        assert arr.grid == tuple(1 for _ in self.gridshape)
        assert arr.blockshape == arr.shape == self.blockshape
        assert arr.dtype == self.dtype


class TestNumpy:
    pass


def test_tensordot():
    blockshape = (4, 1, 1)
    grid = (2, 2, 2)
    a = BlockArray([np.ones(blockshape)] * 8, grid=grid)
    b = BlockArray([np.ones(blockshape)] * 8, grid=grid)
    axes = [(0, 1), (0, 2)]
    c = np.tensordot(a, b, axes)

    assert c.shape == (2, 2)
    assert c.blockshape == (1, 1)
    assert c.grid == (2, 2)
