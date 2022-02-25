import numpy as np
from multimethod import multimethod
from rosnet.helper.macros import implements


@implements("dot")
@multimethod
def dot(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.multi_dot")
@multimethod
def multi_dot(*args, **kwargs):
    raise NotImplementedError()


@implements("vdot")
@multimethod
def vdot(*args, **kwargs):
    raise NotImplementedError()


@implements("inner")
@multimethod
def inner():
    raise NotImplementedError()


@implements("outer")
@multimethod
def outer():
    raise NotImplementedError()


@implements("matmul")
@multimethod
def matmul():
    raise NotImplementedError()


@implements("tensordot")
@multimethod
def tensordot():
    raise NotImplementedError()


@implements("einsum")
@multimethod
def einsum(*args, **kwargs):
    raise NotImplementedError()


@implements("einsum_path")
@multimethod
def einsum_path(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.matrix_power")
@multimethod
def matrix_power(*args, **kwargs):
    raise NotImplementedError()


@implements("kron")
@multimethod
def kron():
    raise NotImplementedError()


@implements("linalg.cholesky")
@multimethod
def cholesky():
    raise NotImplementedError()


@implements("linalg.qr")
@multimethod
def qr():
    raise NotImplementedError()


@implements("linalg.svd")
@multimethod
def svd():
    raise NotImplementedError()


@implements("linalg.eig")
@multimethod
def eig():
    raise NotImplementedError()


@implements("linalg.eigh")
@multimethod
def eigh():
    raise NotImplementedError()


@implements("linalg.eigvals")
@multimethod
def eigvals(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.eigvalsh")
@multimethod
def eigvalsh(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.norm")
@multimethod
def norm():
    raise NotImplementedError()


@implements("linalg.cond")
@multimethod
def cond(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.det")
@multimethod
def det(*args, **kwargs):
    raise NotImplementedError()


@implements("matrix_rank")
@multimethod
def matrix_rank(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.slogdet")
@multimethod
def slogdet(*args, **kwargs):
    raise NotImplementedError()


@implements("trace")
@multimethod
def trace():
    raise NotImplementedError()


@implements("linalg.solve")
@multimethod
def solve(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.tensorsolve")
@multimethod
def tensorsolve(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.lstsq")
@multimethod
def lstsq(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.inv")
@multimethod
def inv(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.pinv")
@multimethod
def pinv(*args, **kwargs):
    raise NotImplementedError()


@implements("linalg.tensorinv")
@multimethod
def tensorinv(*args, **kwargs):
    raise NotImplementedError()
