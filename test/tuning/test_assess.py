import pytest
import numpy as np
from autoray import do
from test.mock import MockArray


class TestTensordot:
    @pytest.mark.parametrize("m", [1, 2, 21, 32768])
    @pytest.mark.parametrize("n", [1, 2, 21, 32768])
    @pytest.mark.parametrize("k", [1, 2, 21, 32768])
    def test_matmul(self, m, n, k):
        a = MockArray((m, k), dtype=np.float64)
        b = MockArray((k, n), dtype=np.float64)
        axes = [(1,), (0, 1)]

        assert do("tensordot", a, b, axes, like="rosnet.tuning.flops") == m * n * k
