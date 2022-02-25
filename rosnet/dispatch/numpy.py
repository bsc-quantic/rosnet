from multimethod import multimethod


@multimethod
def tensordot(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def einsum(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def reshape(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def transpose(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def stack(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def split(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def block(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def zeros_like(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def ones_like(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def full_like(*args, **kwargs):
    raise NotImplementedError()


@multimethod
def empty_like(*args, **kwargs):
    raise NotImplementedError()
