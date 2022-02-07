import numpy as np
from plum import dispatch
from rosnet.helper.macros import implements


@implements(np.tensordot)
@dispatch.abstract
def tensordot(a, b, axes):
    pass


@implements(np.einsum)
@dispatch.abstract
def einsum():
    pass


@implements(np.reshape)
@dispatch.abstract
def reshape():
    pass


@implements(np.transpose)
@dispatch.abstract
def transpose():
    pass


@implements(np.stack)
@dispatch.abstract
def stack():
    pass


@implements(np.split)
@dispatch.abstract
def split():
    pass


@implements(np.block)
@dispatch.abstract
def block():
    pass


@implements(np.zeros_like)
@dispatch.abstract
def zeros_like():
    pass


@implements(np.ones_like)
@dispatch.abstract
def ones_like():
    pass


@implements(np.full_like)
@dispatch.abstract
def full_like():
    pass


@implements(np.empty_like)
@dispatch.abstract
def empty_like():
    pass
