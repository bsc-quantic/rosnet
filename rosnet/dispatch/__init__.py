import numpy as np
from multimethod import multimethod

# custom
@multimethod
def to_numpy(*args, **kwargs):
    raise NotImplementedError()


@to_numpy.register(np.ndarray)
@to_numpy.register(np.generic)
def _(a):
    return a


from .numpy import (
    tensordot,
    einsum,
    reshape,
    transpose,
    stack,
    split,
    block,
    zeros_like,
    ones_like,
    full_like,
    empty_like,
    cumsum,
    count_nonzero,
)

from . import linalg

# import ufuncs
__ufuncs = filter(lambda x: isinstance(x[1], np.ufunc), {attr: getattr(np, attr) for attr in np.__dict__}.items())

for attr, ufunc in __ufuncs:
    globals()[attr] = ufunc
