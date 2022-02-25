import numpy as np
from multimethod import multimethod


@multimethod
def dot(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def multi_dot(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def vdot(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def inner():
    raise NotImplementedError()


@multimethod
def outer():
    raise NotImplementedError()


@multimethod
def matmul():
    raise NotImplementedError()


@multimethod
def tensordot():
    raise NotImplementedError()


@multimethod
def einsum(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def einsum_path(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def matrix_power(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def kron():
    raise NotImplementedError()


@multimethod
def cholesky():
    raise NotImplementedError()


@multimethod
def qr():
    raise NotImplementedError()


@multimethod
def svd():
    raise NotImplementedError()


@multimethod
def eig():
    raise NotImplementedError()


@multimethod
def eigh():
    raise NotImplementedError()


@multimethod
def eigvals(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def eigvalsh(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def norm():
    raise NotImplementedError()


@multimethod
def cond(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def det(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def matrix_rank(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def slogdet(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def trace():
    raise NotImplementedError()


@multimethod
def solve(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def tensorsolve(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def lstsq(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def inv(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def pinv(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def tensorinv(*args, **kwargs):
    raise NotImplementedError()
