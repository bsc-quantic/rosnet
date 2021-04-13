import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import Type, Depth, IN, INOUT, COLLECTION_INOUT, COLLECTION_IN
from dislib_tensor.utils import prod


# NOTE np.transpose returns a view, does not perform the transformation
@ task(block=INOUT)
def block_transpose(block: np.ndarray, permutator):
    block = np.transpose(block, permutator)


@ task(blocks={Type: COLLECTION_INOUT, Depth: 1})
def block_merge_split(blocks, block_shape, a):
    # merge blocks
    superblock = np.block(blocks).reshape((-1, len(blocks)), order='F')
    superblock = np.reshape(superblock, block_shape + [len(blocks)], order='F')

    # transpose indexes
    permutator = list(range(len(block_shape) + 1))
    permutator[a], permutator[-1] = permutator[-1], permutator[a]
    superblock = np.transpose(superblock, permutator)

    # split blocks
    blocks = np.split(
        superblock, superblock.shape[-1], axis=len(superblock.shape)-1)
    blocks = [np.reshape(block, (prod(block.shape),), order='F')
              for block in blocks]


@task(returns=np.array)
def block_full(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)


@task(returns=np.array)
def block_rand(shape):
    return np.random.random_sample(shape)


@task(block=INOUT)
def block_setitem(block, idx, value):
    block[idx] = value


@task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def block_tensordot(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@task(block=IN, returns=np.ndarray)
def block_pass(block):
    return block


@task(block=IN)
def block_getitem(block: np.ndarray, idx: tuple):
    return block[idx]
