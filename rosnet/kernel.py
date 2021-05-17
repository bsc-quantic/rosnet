import os
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import Type, Depth, IN, INOUT, COLLECTION_INOUT, COLLECTION_IN, COLLECTION_OUT


def __numpy_transpose(a, axes=None):
    return np.asfortranarray(np.transpose(a, axes))


# defaults
__backend_transpose = os.environ.get('ROSNET_BACKEND_TRANSPOSE', 'numpy')

if __backend_transpose == 'numpy':
    transpose = __numpy_transpose
else:
    try:
        transpose = __import__(__backend_transpose,
                               fromlist=('transpose')).transposes
    except ImportError:
        transpose = __numpy_transpose
        __backend_transpose = 'numpy'


@task(block=IN, returns=np.ndarray)
def block_transpose(block: np.ndarray, permutator):
    return transpose(block, permutator)


@task(block=IN, returns=np.ndarray)
def block_reshape(block: np.ndarray, shape: tuple):
    return block.reshape(shape, order='F')


@task(block=IN, returns={Type: COLLECTION_OUT, Depth: 1})
def block_split(block: np.ndarray, n: int, axis: int):
    return map(lambda x: x.copy(), np.split(block, n, axis))


@task(blocks=COLLECTION_IN, returns=np.ndarray)
def block_merge(blocks, axis: int):
    return np.stack(blocks, axis)


@task(returns=np.array)
def block_full(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)


@task(returns=np.array)
def block_rand(shape):
    return np.asfortranarray(np.random.random_sample(shape))


@task(block=INOUT)
def block_setitem(block, idx, value):
    block[idx] = value


@task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def block_tensordot(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@task(block=IN, returns=np.ndarray)
def block_copy(block):
    return np.copy(block)


@task(block=IN, returns=np.ndarray)
def block_pass(block):
    return block


@task(block=IN)
def block_getitem(block: np.ndarray, idx: tuple):
    return block[idx]


@task(returns=np.ndarray)
def block_kron(a: np.ndarray, b: np.ndarray):
    return np.kron(a, b)
