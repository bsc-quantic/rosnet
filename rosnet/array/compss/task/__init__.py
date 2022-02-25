import rosnet.task.tensordot
from .init import full, rand, upload
from .util import reshape, setitem, getitem, copy
from .transpose import transpose
from .svd import svdmatrix_async, svdmatrix_async_nested
from .slicing import split, merge
from .kron import kron
from .functional import operate, ioperate
