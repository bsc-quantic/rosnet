import pytest
from rosnet.kernel import *
from pycompss.api.api import compss_wait_on


@pytest.mark.skip(reason="does nothing")
def test_block_pass():
    pass


@pytest.mark.skip(reason="untestable")
def test_block_rand():
    pass


@pytest.mark.parametrize(("shape,value,dtype"), [
    ((1), 7, int),
    ((2, 2, 2), 1, int)
])
def test_block_full(shape, value, dtype):
    block = block_full(shape, value, dtype)
    assert all(i == value for i in block.flat)


@pytest.mark.parametrize(("shape,default,key"), [
    ((1), 7, (0)),
    ((2, 2, 2), 1, (1, 0, 1))
])
def test_block_getitem(shape, default, key):
    block = block_full(shape, default, type(default))
    assert block[key] == default


@pytest.mark.parametrize(("shape,default,key,value"), [
    ((1), 7, (0), 1),
    ((2, 2, 2), 1, (1, 0, 1), 2)
])
def test_block_setitem(shape, default, key, value):
    block = block_full(shape, default, type(default))

    assert block[key] != value
    block_setitem(block, key, value)
    assert block[key] == value


@pytest.mark.parametrize(("a,b,axes,check"), [
    (
        [block_full((2, 2, 2, 2), 1, int)],
        [block_full((2, 2), 1, int)],
        0,
        1
    ), (
        [block_full((2, 2, 2, 2), 1, int)],
        [block_full((2, 2), 1, int)],
        1,
        2
    ), (
        [block_full((2, 2, 2, 2), 1, int)],
        [block_full((2, 2), 1, int)],
        2,
        4
    ), (
        [block_full((2, 2, 2, 2), 1, int)],
        [block_full((2, 2), 1, int)],
        ([0], [0]),
        2
    ), (
        [block_full((2, 2, 2, 2), 1, int)],
        [block_full((2, 2), 1, int)],
        ([0], [1]),
        2
    ), (
        [block_full((2, 2, 2, 2), 1, int)],
        [block_full((2, 2), 1, int)],
        ([0, 1], [1, 0]),
        4
    )
])
def test_block_tensordot(a, b, axes, check):
    c = block_tensordot(a, b, axes)
    assert all(i == check for i in c.flat)
