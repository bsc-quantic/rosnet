import pytest
import numpy as np
from pycompss.api.api import compss_wait_on
import rosnet.task as task



@pytest.mark.parametrize(("shape,value,dtype"), [
    ((1), 7, int),
    ((2, 2, 2), 1, int)
])
def test_full(shape, value, dtype):
    block = task.full(shape, value, dtype)
    assert all(i == value for i in block.flat)


@pytest.mark.parametrize(("shape,default,key"), [
    ((1), 7, (0)),
    ((2, 2, 2), 1, (1, 0, 1))
])
def test_getitem(shape, default, key):
    block = task.full(shape, default, type(default))
    assert block[key] == default


@pytest.mark.parametrize(("shape,default,key,value"), [
    ((1), 7, (0), 1),
    ((2, 2, 2), 1, (1, 0, 1), 2)
])
def test_setitem(shape, default, key, value):
    block = task.full(shape, default, type(default))

    assert block[key] != value
    task.setitem(block, key, value)
    assert block[key] == value


@pytest.mark.parametrize(("t,n,axis,a,b"), [
    (
        task.full((2, 2, 2, 2), 1, int), 2, 0,
        task.full((1, 2, 2, 2), 1, int), task.full((1, 2, 2, 2), 1, int)
    ), (
        task.full([2]*12, 1, int), 2, -1,
        task.full([2]*11 + [1], 1, int), task.full([2]*11 + [1], 1, int)
    )
])
def test_split(t, n, axis, a, b):
    at, bt = task.split(t, n, axis)
    assert np.array_equal(a, at)
    assert np.array_equal(b, bt)
    assert at.base == None
    assert bt.base == None
