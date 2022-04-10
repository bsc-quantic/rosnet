from . import tensordot
from .astype import astype_copy
from .count_nonzero import count_nonzero
from .cumsum import cumsum, cumsum_out
from .einsum import einsum, einsum_out
from .functional import ioperate, operate, ufunc_out
from .init import full, rand
from .kron import kron
from .qr import qr_complete, qr_r, qr_raw, qr_reduced
from .slicing import split, stack
from .svd import svd, svd_matrix, svd_vals
from .transpose import transpose, transpose_inplace
from .util import copy, getitem, reshape, reshape_inplace, setitem
