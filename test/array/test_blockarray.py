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

    def test_nested_list_of_array(self):
        data = np.empty(self.gridshape, dtype=object)
        it = np.nditer(data, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

        with it:
            for block in it:
                block[()] = np.empty(self.blockshape, dtype=self.dtype)

        arr = BlockArray(data.tolist())

        assert arr.grid == self.gridshape
        assert arr.blockshape == self.blockshape
        assert arr.dtype == self.dtype

    def test_list_of_array(self):
        data = [np.zeros(self.blockshape, dtype=self.dtype)] * prod(self.gridshape)

        arr = BlockArray(data, grid=self.gridshape)

        assert arr.grid == self.gridshape
        assert arr.blockshape == self.blockshape
        assert arr.dtype == self.dtype

    def test_nested_array(self):
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
        data = np.zeros(self.shape, dtype=self.dtype)
        arr = BlockArray(data)

        assert arr.grid == tuple(1 for _ in self.gridshape)
        assert arr.blockshape == arr.shape == self.shape
        assert arr.dtype == self.dtype

    # def test_array(self):
    #     class MockArray:
    #         def __array__(self):
    #             pass

    #         def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #             pass

    #         def __array_function__(self, function, types, args, kwargs):
    #             pass

    #         def shape(self) -> Tuple[int]:
    #             pass

    #         def dtype(self) -> np.dtype:
    #             pass

    #     arr = MockArray()
    #     rn.


class TestNumpy:
    pass
