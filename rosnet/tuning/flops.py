from math import prod
import numpy as np
from plum import dispatch
from rosnet.helper.macros import implements
from rosnet.helper.typing import Array
from rosnet.helper.math import result_shape


# @implements("tensordot", ext="tuning.flops")
# @dispatch.abstract
# def tensordot():
#     pass


@implements("tensordot", ext="tuning.flops")
# @dispatch
def tensordot(a: Array, b: Array, axes) -> int:
    assert all(a.shape[i] == b.shape[j] for i, j in zip(*axes))

    outer_shape = list(result_shape(a.shape, b.shape, axes))
    inner_shape = [a.shape[i] for i in axes[0]]
    return prod(outer_shape + inner_shape)


# @dispatch
# def tensordot(a: List[Array], b: List[Array], axes) -> int:
#     pass


@implements("full", ext="tuning.flops")
def full(shape, fill_value, dtype=None, **kwargs) -> int:
    return prod(shape)


@implements("reshape", ext="tuning.flops")
def reshape(a: Array, shape, **kwargs) -> int:
    # TODO does not change/move data, only the .shape attr. maybe if bad alignment afterwards?
    return 1


@implements("transpose", ext="tuning.flops")
def transpose(a: Array, axes=None, **kwargs) -> int:
    return a.size
