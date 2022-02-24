import pytest
import numpy as np
from rosnet import BlockArray
from rosnet.interface import ArrayConvertable, Array
from test.mock import MockArray

try:
    import pycompss

    incompss = True
except ImportError:
    incompss = False

try:
    import dataclay

    indataclay = True
except ImportError:
    indataclay = False


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

    def test_dataclayarray(self):
        pytest.importorskip("dataclay")
        from rosnet import DataClayArray

        assert issubclass(DataClayArray, ArrayConvertable)


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

    def test_dataclayarray(self):
        pytest.importorskip("dataclay")
        from rosnet import DataClayArray

        assert issubclass(DataClayArray, Array)
