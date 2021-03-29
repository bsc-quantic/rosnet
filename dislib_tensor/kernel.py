import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import Type, Depth, INOUT, COLLECTION_INOUT
from dislib_tensor.utils import prod


@ task(block=INOUT)
def _block_permute(block, shape, a, b):
    rank = len(shape)
    permutator = list(range(rank))
    permutator[a], permutator[b] = b, a

    block = np.reshape(block, shape, order='F')
    block = np.transpose(block, permutator)
    block = np.reshape(block, (prod(shape),), order='F')


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

    # permute indexes
    print("\tpermute indexes:")
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
