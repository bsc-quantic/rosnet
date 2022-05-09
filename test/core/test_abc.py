import pytest
import numpy as np
from rosnet import BlockArray
from rosnet.core.abc import ArrayConvertable, Array
from test.mock import MockArray


class TestArrayConvertableInterface:
    def test_mockarray(self):
        assert issubclass(MockArray, ArrayConvertable)

    def test_ndarray(self):
        assert issubclass(np.ndarray, ArrayConvertable)

    def test_blockarray(self):
        assert issubclass(BlockArray, ArrayConvertable)

    def test_compssarray(self):
        pytest.importorskip("pycompss")
        from rosnet import COMPSsArray

        assert issubclass(COMPSsArray, ArrayConvertable)


class TestArrayInterface:
    def test_mockarray(self):
        assert issubclass(MockArray, Array)

    def test_ndarray(self):
        assert issubclass(np.ndarray, Array)

    def test_blockarray(self):
        assert issubclass(BlockArray, Array)

    def test_compssarray(self):
        pytest.importorskip("pycompss")
        from rosnet import COMPSsArray

        assert issubclass(COMPSsArray, Array)
