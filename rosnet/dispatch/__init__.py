import numpy as np
from multimethod import multimethod

# custom
@multimethod
def to_numpy(*args, **kwargs):
    raise NotImplementedError()


@to_numpy.register
def _(a: np.ndarray):
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
)

from . import linalg
