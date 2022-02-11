import pytest
from typing import Tuple
import numpy as np
from rosnet.helper.math import recurse, nest_level, measure_shape


class MockArray:
    def __init__(self, shape, **kwargs):
        self.__shape = shape
        self.__dtype = kwargs.get("dtype", np.generic)

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
        return self.__shape

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)


class TestRecurseNestLevel:
    def test_ndarray(self):
        arr = np.empty((1,))

        assert nest_level(arr) == 0

    def test_array(self):
        arr = MockArray((1,), dtype=np.generic)

        assert nest_level(arr) == 0

    def test_ndarray_ndarray(self):
        arr = np.empty((1,), dtype=object)
        arr.flat[0] = np.empty((1,))

        assert nest_level(arr) == 1

    def test_ndarray_array(self):
        arr = np.empty((1,), dtype=object)
        arr.flat[0] = MockArray((1,), dtype=np.generic)

        assert nest_level(arr) == 1

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_nested_list_of_ndarray(self, level):
        arr = np.empty(tuple([1] * level), dtype=object)
        arr.flat[0] = np.zeros((1,))

        assert nest_level(arr.tolist()) == level

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_nested_list_of_array(self, level):
        arr = np.empty(tuple([1] * level), dtype=object)
        arr.flat[0] = MockArray((1,))

        assert nest_level(arr.tolist()) == level


class TestRecurseShapeTracker:
    def test_ndarray(self):
        arr = np.empty((1,))

        assert measure_shape(arr) == ()

    def test_array(self):
        arr = MockArray((1,), dtype=np.generic)

        assert measure_shape(arr) == ()

    def test_ndarray_ndarray(self):
        arr = np.empty((1,), dtype=object)
        arr.flat[0] = np.empty((1,))

        assert measure_shape(arr) == (1,)

    def test_ndarray_array(self):
        arr = np.empty((1,), dtype=object)
        arr.flat[0] = MockArray((1,), dtype=np.generic)

        assert measure_shape(arr) == (1,)

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_nested_list_of_ndarray(self, level):
        arr = np.empty(tuple([1] * level), dtype=object)
        arr.flat[0] = np.zeros((1,))

        assert measure_shape(arr.tolist()) == tuple([1] * level)

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_nested_list_of_array(self, level):
        arr = np.empty(tuple([1] * level), dtype=object)
        arr.flat[0] = MockArray((1,))

        assert measure_shape(arr.tolist()) == tuple([1] * level)
