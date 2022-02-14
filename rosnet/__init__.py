__version__ = "0.1"

from rosnet.array import *
from rosnet.interface import (
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

# NumPy methods whose output type cannot be inferred. defaults to BlockArray.
# for other kinds of arrays, use `autoray.do(..., like="rosnet.CLASSNAME")`
from rosnet.array.block import (
    zeros,
    ones,
    full,
    rand,
)

from rosnet.linalg import (
    dot,
    vdot,
    inner,
    outer,
    matmul,
    tensordot,
    einsum,
    einsum_path,
    kron,
    trace,
)
