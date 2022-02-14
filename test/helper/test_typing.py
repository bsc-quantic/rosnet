import pytest
import numpy as np
from rosnet import BlockArray
from rosnet.helper.typing import SupportsArray, Array
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


class TestSupportsArrayProtocol:
    def test_mockarray(self):
        assert issubclass(MockArray, SupportsArray)

    def test_ndarray(self):
        assert issubclass(np.ndarray, SupportsArray)

    def test_blockarray(self):
        assert issubclass(BlockArray, SupportsArray)

    def test_compssarray(self):
        pytest.importorskip("pycompss")
        from rosnet import COMPSsArray

        assert issubclass(COMPSsArray, SupportsArray)

    def test_dataclayarray(self):
        pytest.importorskip("dataclay")
        from rosnet import DataClayArray

        assert issubclass(DataClayArray, SupportsArray)


class TestArrayProtocol:
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
