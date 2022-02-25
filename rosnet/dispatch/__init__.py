from multimethod import multimethod

# custom
@multimethod
def to_numpy(*args, **kwargs):
    raise NotImplementedError()


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
)
