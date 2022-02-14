import numpy as np
from plum import dispatch
from rosnet.helper.macros import implements


@implements("dot")
def dot(*args, **kwargs):
    return NotImplemented


@implements("linalg.multi_dot")
def multi_dot(*args, **kwargs):
    return NotImplemented


@implements("vdot")
def vdot(*args, **kwargs):
    return NotImplemented


@implements("inner")
@dispatch.abstract
def inner():
    pass


@implements("outer")
@dispatch.abstract
def outer():
    pass


@implements("matmul")
@dispatch.abstract
def matmul():
    pass


@implements("tensordot")
@dispatch.abstract
def tensordot():
    pass


@implements("einsum")
def einsum(*args, **kwargs):
    return NotImplemented


@implements("einsum_path")
def einsum_path(*args, **kwargs):
    return NotImplemented


@implements("linalg.matrix_power")
def matrix_power(*args, **kwargs):
    return NotImplemented


@implements("kron")
@dispatch.abstract
def kron():
    pass


@implements("linalg.cholesky")
@dispatch.abstract
def cholesky():
    pass


@implements("linalg.qr")
@dispatch.abstract
def qr():
    pass


@implements("linalg.svd")
@dispatch.abstract
def svd():
    pass


@implements("linalg.eig")
@dispatch.abstract
def eig():
    pass


@implements("linalg.eigh")
@dispatch.abstract
def eigh():
    pass


@implements("linalg.eigvals")
def eigvals(*args, **kwargs):
    return NotImplemented


@implements("linalg.eigvalsh")
def eigvalsh(*args, **kwargs):
    return NotImplemented


@implements("linalg.norm")
@dispatch.abstract
def norm():
    pass


@implements("linalg.cond")
def cond(*args, **kwargs):
    return NotImplemented


@implements("linalg.det")
def det(*args, **kwargs):
    return NotImplemented


@implements("matrix_rank")
def matrix_rank(*args, **kwargs):
    return NotImplemented


@implements("linalg.slogdet")
def slogdet(*args, **kwargs):
    return NotImplemented


@implements("trace")
@dispatch.abstract
def trace():
    pass


@implements("linalg.solve")
def solve(*args, **kwargs):
    return NotImplemented


@implements("linalg.tensorsolve")
def tensorsolve(*args, **kwargs):
    return NotImplemented


@implements("linalg.lstsq")
def lstsq(*args, **kwargs):
    return NotImplemented


@implements("linalg.inv")
def inv(*args, **kwargs):
    return NotImplemented


@implements("linalg.pinv")
def pinv(*args, **kwargs):
    return NotImplemented


@implements("linalg.tensorinv")
def tensorinv(*args, **kwargs):
    return NotImplemented
