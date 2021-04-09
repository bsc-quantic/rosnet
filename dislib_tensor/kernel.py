import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import Type, Depth, IN, INOUT, COLLECTION_INOUT, COLLECTION_IN
from dislib_tensor.utils import prod


# NOTE np.transpose returns a view, does not perform the transformation
@ task(block=INOUT)
def _block_transpose(block: np.ndarray, permutator):
    block = np.transpose(block, permutator)


@ task(blocks={Type: COLLECTION_INOUT, Depth: 1})
def _block_merge_split(blocks, block_shape, a):
    print("[DEBUG] _block_merge_split:")
    print("\ta = %s" % str(a))
    print("\tblock_shape = %s" % block_shape)
    print("\tlen(blocks) = %s" % len(blocks))

    # merge blocks
    print("\tmerge blocks:")
    superblock = np.block(blocks).reshape((-1, len(blocks)), order='F')
    print("\t\t1 - shape(superblock) = %s" % str(superblock.shape))
    superblock = np.reshape(superblock, block_shape + [len(blocks)], order='F')
    print("\t\t2 - shape(superblock) = %s" % str(superblock.shape))

    # transpose indexes
    print("\ttranspose indexes:")
    permutator = list(range(len(block_shape) + 1))
    permutator[a], permutator[-1] = permutator[-1], permutator[a]
    print("\t\tpermutator = %s" % str(permutator))
    superblock = np.transpose(superblock, permutator)
    print("\t\tshape(superblock) = %s" % str(superblock.shape))

    # split blocks
    print("\tsplit blocks:")
    blocks = np.split(
        superblock, superblock.shape[-1], axis=len(superblock.shape)-1)
    blocks = [np.reshape(block, (prod(block.shape),), order='F')
              for block in blocks]
    print("\t\tlen(blocks) = %s" % str(len(blocks)))
    print("\t\tblocks[0].shape = %s" % str(blocks[0].shape))


@task(returns=np.array)
def _block_full(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)


@task(returns=np.array)
def _block_rand(shape):
    return np.random.random_sample(shape)


@task(block=INOUT)
def _block_set_value(block, idx, value):
    block[idx] = value


@task(a={Type: COLLECTION_IN, Depth: 1}, b={Type: COLLECTION_IN, Depth: 1}, returns=np.ndarray)
def _block_tensordot(a, b, axes):
    return sum(np.tensordot(ba, bb, axes) for ba, bb in zip(a, b))


@task(block=IN, returns=np.ndarray)
def _block_pass_block(block):
    return block


@task(block=IN)
def _block_getitem(block: np.ndarray, idx: tuple):
    return block[idx]
