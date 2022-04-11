__version__ = "0.4.dev"

from rosnet.array import *
from rosnet.dispatch import *
import rosnet.dispatch.linalg as linalg

# NumPy methods whose output type cannot be inferred. defaults to BlockArray.
# for other kinds of arrays, use `autoray.do(..., like="rosnet.CLASSNAME")`
from rosnet.array.block import (
    zeros,
    ones,
    full,
    rand,
)

from rosnet.extra import *
