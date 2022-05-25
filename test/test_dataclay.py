import pytest
import numpy as np

model = pytest.importorskip("rosnet_dclaymodel.model")


class TestDataClayBlock:
    def test_init(self):
        arr = np.random.rand(2, 2, 2, 2)
        block = model.DataClayBlock(arr)
        block.make_persistent()

        retrieved_arr = block.data
        assert np.array_equal(retrieved_arr, arr)

    def test_array(self):
        arr = np.random.rand(2, 2, 2, 2)
        block = model.DataClayBlock(arr)
        block.make_persistent()

        retrieved_arr = np.array(block)
        assert np.array_equal(retrieved_arr, arr)
