import numpy as np
from pycompss.api.task import task
from pycompss.api.constraint import constraint


@task(returns=np.array)
def full_0(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)

@constraint(memory_size="1")
@task(returns=np.array)
def full_1(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)

@constraint(memory_size="2")
@task(returns=np.array)
def full_2(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)

@constraint(memory_size="4")
@task(returns=np.array)
def full_4(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)

@constraint(memory_size="8")
@task(returns=np.array)
def full_8(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)

@constraint(memory_size="16")
@task(returns=np.array)
def full_16(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)

@constraint(memory_size="32")
@task(returns=np.array)
def full_32(shape, value, dtype, order='F'):
    return np.full(shape, value, dtype, order)


full = full_0


@task(returns=np.array)
def rand(shape, order='F'):
    return np.asarray(np.random.random_sample(shape), order=order)


@task(returns=np.ndarray)
def upload(block):
    return block


@task(returns=np.ndarray)
def identity(block_shape, n, i, j, dtype):
    block = np.zeros(block_shape, dtype)

    diag = np.intersect1d(*[np.arange(idx * bs, min(n, (idx+1)*bs))
                            for idx, bs in zip([i, j], block_shape)])

    i_ones, j_ones = [diag - idx * bs for idx, bs in zip([i, j], block_shape)]

    block[i_ones, j_ones] = 1

    return block
