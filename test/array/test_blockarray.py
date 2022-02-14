import pytest
from typing import Tuple
from math import prod
import numpy as np
from rosnet import BlockArray
from test.mock import MockArray


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

    def test_nested_list_of_array(self):
        data = np.empty(self.gridshape, dtype=object)
        it = np.nditer(data, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

        with it:
            for block in it:
                block[()] = MockArray(self.blockshape, dtype=self.dtype)

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

    def test_list_of_array(self):
        data = [MockArray(self.blockshape, dtype=self.dtype)] * prod(self.gridshape)

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

    def test_nested_array(self):
        data = np.empty(self.gridshape, dtype=object)
        it = np.nditer(data, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

        with it:
            for block in it:
                block[()] = MockArray(self.blockshape, dtype=self.dtype)

        arr = BlockArray(data)
        print(f"[out]arr.grid={arr.grid}")
        print(f"[out]arr.blockshape={arr.blockshape}")

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
        data = MockArray(self.blockshape, dtype=self.dtype)
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
