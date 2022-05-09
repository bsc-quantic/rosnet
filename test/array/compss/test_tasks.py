from math import prod
import pytest
import numpy as np
from autoray import do

pycompss = pytest.importorskip("pycompss")
from rosnet import COMPSsArray


@pytest.mark.skip()
class TestTensordot:
    def test_tensordot(self):
        pass

    def test_sequential(self):
        pass

    def test_commutative(self):
        pass


class TestTranspose:
    a = np.random.rand(2, 1, 4, 8)
    axes = [
        # identity
        tuple(range(a.ndim)),
        list(range(a.ndim)),
        # singleton dim
        (1, 0, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 2, 1),
        # reverse
        (3, 2, 1, 0),
        # random
        (2, 1, 3, 0),
    ]

    @pytest.mark.parametrize("axes", [None] + axes)
    def test_transpose(self, axes):
        a = COMPSsArray(self.a)
        b = np.array(do("transpose", a, axes))

        assert np.array_equal(b, np.transpose(self.a, axes))

    @pytest.mark.parametrize("axes", axes)
    def test_transpose_expansion(self, axes):
        a = COMPSsArray(self.a)
        b = np.array(a.transpose(*axes))

        assert np.array_equal(b, self.a.transpose(*axes))


class TestReshape:
    a = np.random.rand(2, 1, 4, 8)

    @pytest.mark.parametrize(
        "shape",
        [
            # identity
            a.shape,
            # vector
            (prod(a.shape),),
            # vector passed by int
            prod(a.shape),
            # shrink dims
            (2, 4, 8),
            (2, 32),
            (8, 8),
            # expand dims
            (2, 1, 2, 2, 8),
            (2, 1, 2, 2, 2, 4),
            (2, 1, 2, 2, 2, 2, 2),
            # add singleton dims
            (*a.shape, 1, 1),
            (1, 1, *a.shape),
            # -1 inference
            (-1, *a.shape[0:1]),
            (*a.shape[0:1], -1),
            (*a.shape[0:1], -1, 2),
            # TODO (2, -1, 4, -1, 2), must fail
        ],
    )
    def test_reshape(self, shape):
        a = COMPSsArray(self.a)
        b = np.array(do("reshape", a, shape))

        assert np.array_equal(b, np.reshape(self.a, shape))


@pytest.mark.skip()
class TestSlicing:
    pass


@pytest.mark.skip()
class TestSVD:
    pass


@pytest.mark.skip()
class TestQR:
    pass
