import pytest
import numpy as np
import rosnet as rn


def test_block_tensordot():
    zeros = np.zeros((4, 4))
    a = rn.array(zeros + 1, blockshape=(2, 2))
    b = rn.array(zeros + 2, blockshape=(2, 2))
    axes = [(0,), (0,)]
    c = np.tensordot(a, b, axes)


def test_block_numpy_asarray():
    zeros = np.zeros((4, 4))
    a = rn.array(zeros, blockshape=(2, 2))
    c = np.asarray(a)
