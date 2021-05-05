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
    ), (
        [block_full((2, 2, 2, 2), 0, int)],
        [block_full((2, 2), 1, int)],
        ([0, 1], [1, 0]),
        0
    ), (
        [block_full((2, 2, 2, 2), 1, int)],
        [np.eye(2, 2, dtype=int, order='F')],
        ([1], [0]),
        1
    )
])
def test_block_tensordot(a, b, axes, check):
    c = block_tensordot(a, b, axes)
    assert all(i == check for i in c.flat)


@pytest.mark.parametrize(("a,b,perm"), [
    (
        [
            [1, 0],
            [0, 1]
        ],
        [
            [1, 0],
            [0, 1]
        ],
        (0, 1)
    ), (
        [
            [1, 0],
            [0, 1]
        ],
        [
            [1, 0],
            [0, 1]
        ],
        (1, 0)
    ), (
        [
            [1, 2],
            [3, 4]
        ],
        [
            [1, 2],
            [3, 4]
        ],
        (0, 1)
    ), (
        [
            [1, 2],
            [3, 4]
        ],
        [
            [1, 3],
            [2, 4]
        ],
        (1, 0)
    )
])
def test_block_transpose(a, b, perm):
    at = block_transpose(a, perm)
    assert (at == b).all()


@pytest.mark.parametrize(("t,n,axis,a,b"), [
    (
        block_full((2, 2, 2, 2), 1, int), 2, 0,
        block_full((1, 2, 2, 2), 1, int), block_full((1, 2, 2, 2), 1, int)
    ), (
        block_full([2]*12, 1, int), 2, -1,
        block_full([2]*11 + [1], 1, int), block_full([2]*11 + [1], 1, int)
    )
])
def test_block_split(t, n, axis, a, b):
    at, bt = block_split(t, n, axis)
    assert np.array_equal(a, at)
    assert np.array_equal(b, bt)


