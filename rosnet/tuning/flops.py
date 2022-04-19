from math import prod

import numpy as np
from rosnet.core.abc import Array
from rosnet.core.util import result_shape


def tensordot(a: Array, b: Array, axes) -> int:
    assert all(a.shape[i] == b.shape[j] for i, j in zip(*axes))

    outer_shape = list(result_shape(a.shape, b.shape, axes))
    inner_shape = [a.shape[i] for i in axes[0]]
    return prod(outer_shape + inner_shape)


def full(shape, fill_value, dtype=None, **kwargs) -> int:
    return prod(shape)


def reshape(a: Array, shape, **kwargs) -> int:
    # TODO does not change/move data, only the .shape attr. maybe if bad alignment afterwards?
    return 1


def transpose(a: Array, axes=None, **kwargs) -> int:
    return a.size
