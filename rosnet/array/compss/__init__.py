import functools
import logging
from copy import deepcopy
from math import prod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from opt_einsum.parser import find_output_shape, parse_einsum_input
from pycompss.api.api import compss_delete_object, compss_wait_on
from pycompss.runtime.management.classes import Future as COMPSsFuture
from rosnet import dispatch as dispatcher
from rosnet import tuning
from rosnet.array.block import BlockArray
from rosnet.array.maybe import MaybeArray
from rosnet.core.interface import Array, ArrayConvertable, AsyncArray
from rosnet.core.log import log_args
from rosnet.core.macros import todo
from rosnet.core.util import isunique, result_shape
from rosnet.core.mixin import ArrayFunctionMixin

from . import task

logger = logging.getLogger(__name__)

try:
    from rosnet.array.compss.dataclay import DataClayBlock

    DATACLAY = True

except ImportError:
    DATACLAY = False

# TODO special variation for in-place functions? keep np.reshape/transpose/... as non-modifying -> create new COMPSsArray/BlockArray
# TODO support more properties of ndarray
class COMPSsArray(np.lib.mixins.NDArrayOperatorsMixin, ArrayFunctionMixin):
    """Reference to a `numpy.ndarray` managed by COMPSs.

    Unlike a `numpy.ndarray`, a `COMPSsArray` is mutable and does not return views. As such, the following methods may act in-place and return themselves:
    - `reshape`
    - `transpose`
    """

    data: Union[Array, COMPSsFuture]

    # pylint: disable=protected-access
    def __init__(self, arr, **kwargs):
        self.__init_dispatch(arr, **kwargs)

    @functools.singledispatchmethod
    def __init_dispatch(self, arr, **kwargs):
        self.data = arr
        self._shape = kwargs.get("shape", None) or arr.shape
        self.__dtype = kwargs.get("dtype", None) or arr.dtype

        assert isinstance(self.dtype, (np.dtype, type))
        self.__dtype = np.dtype(self.__dtype)

    @__init_dispatch.register
    def _(self, arr: ArrayConvertable, **kwargs):
        "Constructor for generic arrays."
        self.data = np.array(arr)
        self._shape = arr.shape
        self.__dtype = arr.dtype

    @__init_dispatch.register
    def _(self, arr: np.generic, **kwargs):
        "Constructor for scalars."
        self.data = arr
        self._shape = ()
        self.__dtype = arr.dtype

    @__init_dispatch.register
    def _(self, arr: COMPSsFuture, **kwargs):
        "Constructor for future result of COMPSs tasks."
        self.data = arr
        self._shape = kwargs["shape"]
        self.__dtype = kwargs["dtype"]

    def __del__(self):
        logger.debug(f"id={id(self)}, self={self}")
        if isinstance(self.data, COMPSsFuture):
            compss_delete_object(self.data)
        elif DATACLAY:
            if isinstance(self.data, DataClayBlock):
                self.data.session_detach()  # TODO is this call ok?

    def __str__(self) -> str:
        return f"COMPSsArray<data=id({id(self.data)}), shape={self.shape}, dtype={self.dtype}>"

    def __repr__(self) -> str:
        return f"COMPSsArray<id={id(self)}, data=id({id(self.data)}), shape={self.shape}, dtype={self.dtype}>"

    @log_args(logger)
    def __getitem__(self, idx) -> COMPSsFuture:
        return compss_wait_on(task.getitem(self.data, idx))

    @log_args(logger)
    def __setitem__(self, key, value):
        task.setitem(self.data, key, value)

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @shape.setter
    def _(self, shape: Tuple[int]):
        if prod(shape) != prod(self.shape):
            raise ValueError("number of elements of new shape does not match")

        self._shape = shape
        task.reshape_inplace(self.data, shape)

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    @log_args(logger)
    def __deepcopy__(self, memo):
        if isinstance(self.data, COMPSsFuture):
            ref = task.copy(self.data)
        elif DATACLAY:
            if isinstance(self.data, DataClayBlock):
                ref = self.data.dc_clone()
        else:
            ref = deepcopy(self.data)
        return COMPSsArray(ref, shape=self.shape, dtype=self.dtype)

    @log_args(logger)
    def __array__(self) -> np.ndarray:
        return np.array(dispatcher.to_numpy(self.data))

    @log_args(logger)
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if ufunc.nin > 2:
            return NotImplemented

        # get COMPSs reference if COMPSsArray
        inputs_unwrap = [arg.data if isinstance(arg, AsyncArray) else arg for arg in inputs]

        out = kwargs.pop("out", None)
        if out is not None:
            out = tuple(i.data for i in out)
        inplace = out is not None

        # 'at' operates in-place
        if method == "at":
            if not np.can_cast(inputs[1], inputs[0], casting="safe"):
                return NotImplemented
            task.ufunc_out(out, ufunc, *inputs_unwrap, **kwargs)

        # '__call__', 'outer'
        elif method in "__call__":
            if inplace:
                types = [i.dtype if hasattr(i, "dtype") else i for i in inputs]
                if not np.can_cast(types[1], types[0], casting="safe"):
                    return NotImplemented
                task.ufunc_out(out, ufunc, *inputs_unwrap, **kwargs)
                return self
            else:
                ref = task.operate(ufunc, *inputs_unwrap, **kwargs)
                dtype = np.result_type(*(i.dtype if hasattr(i, "dtype") else i for i in inputs))

                return COMPSsArray(ref, shape=self.shape, dtype=dtype)

        elif method == "outer":
            if inplace:
                return NotImplemented
            else:
                ref = task.operate(ufunc, *inputs_unwrap, **kwargs)
                shape = functools.reduce(tuple.__add__, (i.shape for i in inputs))
                dtype = np.result_type(*(i.dtype for i in inputs))
                return COMPSsArray(ref, shape=shape, dtype=dtype)

        # 'reduce', 'accumulate', 'reduceat' not supported yet
        else:
            return NotImplemented

    @log_args(logger)
    def astype(self, dtype: np.dtype, order="K", casting="unsafe", subok=True, copy=True) -> "COMPSsArray":
        # TODO support order, subok
        if not copy:
            raise NotImplementedError()

        ref = task.astype_copy(self.data, dtype=dtype, order=order, casting=casting, subok=subok)

        return COMPSsArray(ref, shape=self.shape, dtype=dtype)

    def reshape(self, shape, order="C") -> "COMPSsArray":
        return dispatcher.reshape[(COMPSsArray,)](self, shape, order=order)

    def transpose(self, *axes):
        return dispatcher.transpose[(COMPSsArray,)](self, axes=axes)

    @property
    def T(self) -> "COMPSsArray":
        return self.transpose()

    def conj(self) -> "COMPSsArray":
        # redirect execution to __array_ufunc__
        return np.conj(self)  # type: ignore


# COMPSsArray is an async array
AsyncArray.register(COMPSsArray)


@dispatcher.to_numpy.register
@log_args(logger)
def _(arr: COMPSsFuture):
    return compss_wait_on(arr)


@dispatcher.to_numpy.register
@log_args(logger)
def to_numpy(arr: COMPSsArray):
    return dispatcher.to_numpy(arr.data)


@dispatcher.to_numpy.register
@log_args(logger)
def _(arr: BlockArray[COMPSsArray]):
    blocks = np.empty_like(arr.data, dtype=object)
    it = np.nditer(
        arr.data,
        flags=["refs_ok", "multi_index"],
        op_flags=["readonly"],
        op_axes=[tuple(range(arr.ndim))],
    )
    with it:
        for x in it:
            blocks[it.multi_index] = dispatcher.to_numpy(x[()])
    return np.block(blocks.tolist())


@log_args(logger)
def zeros(shape, dtype=None, order="C") -> COMPSsArray:
    return full(shape, 0, dtype=dtype, order=order)


@log_args(logger)
def ones(shape, dtype=None, order="C") -> COMPSsArray:
    return full(shape, 1, dtype=dtype, order=order)


@log_args(logger)
def full(shape, fill_value, dtype=None, order="C") -> COMPSsArray:
    ref = task.full(shape, fill_value, dtype=dtype, order=order)
    return COMPSsArray(ref, shape=shape, dtype=dtype or np.dtype(type(fill_value)))


@dispatcher.zeros_like.register
@log_args(logger)
def zeros_like(a: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> Union[np.ndarray, COMPSsArray]:
    if subok:
        return zeros(shape or a.shape, dtype=dtype or a.dtype, order=order)
    else:
        return np.zeros(shape or a.shape, dtype=dtype or a.dtype, order=order)


@dispatcher.ones_like.register
@log_args(logger)
def ones_like(a: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> Union[np.ndarray, COMPSsArray]:
    if subok:
        return ones(shape or a.shape, dtype=dtype or a.dtype, order=order)
    else:
        return np.ones(shape or a.shape, dtype=dtype or a.dtype, order=order)


@dispatcher.full_like.register
@log_args(logger)
def full_like(a: COMPSsArray, fill_value, dtype=None, order="K", subok=True, shape=None) -> Union[np.ndarray, COMPSsArray]:
    if subok:
        return full(shape or a.shape, fill_value, dtype=dtype or a.dtype, order=order)
    else:
        return np.full(shape or a.shape, fill_value, dtype=dtype or a.dtype, order=order)


@dispatcher.empty_like.register
@log_args(logger)
def empty_like(prototype: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@dispatcher.reshape.register
@log_args(logger)
def reshape(a: COMPSsArray, shape, order="C", inplace=False):
    a = a if inplace else deepcopy(a)

    # reshape to 1-D array
    if isinstance(shape, int):
        shape = (shape,)

    # infer shape dimensions
    elif -1 in shape:
        assert sum(1 if d == -1 else 0 for d in shape) <= 1

        inferred_value = -prod(a.shape) // prod(shape)
        shape = tuple(inferred_value if d == -1 else d for d in shape)

    assert prod(a.shape) == prod(shape)

    if inplace:
        # TODO support order
        task.reshape_inplace(a.data, shape)
        return a
    else:
        ref = task.reshape(a.data, shape, order)
        return COMPSsArray(ref, shape=shape, dtype=a.dtype)


@dispatcher.transpose.register
@log_args(logger)
def transpose(a: COMPSsArray, axes=None, inplace=False):
    # case: reverse axes
    if axes is None:
        axes = range(a.ndim)[::-1]

    # case: n-ints
    elif isinstance(axes, Sequence) and all(isinstance(i, int) for i in axes):
        if set(range(a.ndim)) != set(axes):
            raise ValueError(f"axes don't match array: axes={axes}")

    # case: tuple[int,...]
    elif isinstance(axes, Sequence) and len(axes) == 1 and isinstance(axes[0], Sequence):
        axes = axes[0]
        if set(range(a.ndim)) != set(axes):
            raise ValueError(f"axes don't match array: axes={axes}")

    else:
        raise ValueError(f"axes don't match array: axes={axes}")

    shape = tuple(a.shape[i] for i in axes)
    if inplace:
        ref = a.data
        task.transpose_inplace(ref, axes)

        # fix inplace reshape
        a._shape = shape
        return a

    else:
        ref = task.transpose(a.data, axes)
        return COMPSsArray(ref, shape=shape, dtype=a.dtype)


@todo
@dispatcher.stack.register
@log_args(logger)
def stack(arrays: Sequence[COMPSsArray], axis=0, out=None) -> COMPSsArray:
    pass


@todo
@dispatcher.split.register
@log_args(logger)
def split(array: COMPSsArray, indices_or_sections, axis=0) -> Sequence[COMPSsArray]:
    pass


@dispatcher.tensordot.register(COMPSsArray, ArrayConvertable)
@dispatcher.tensordot.register(ArrayConvertable, COMPSsArray)
def tensordot(a: Union[COMPSsArray, ArrayConvertable], b: Union[COMPSsArray, ArrayConvertable], axes):
    a = a if isinstance(a, COMPSsArray) else COMPSsArray(a)
    b = b if isinstance(b, COMPSsArray) else COMPSsArray(b)
    return dispatcher.tensordot[(COMPSsArray, COMPSsArray)](a, b, axes)


@dispatcher.tensordot.register
@log_args(logger)
def tensordot(a: COMPSsArray, b: COMPSsArray, axes) -> COMPSsArray:
    dtype = np.result_type(a.dtype, b.dtype)
    shape = result_shape(a.shape, b.shape, axes)

    ref = task.tensordot.tensordot(a.data, b.data, axes)
    return COMPSsArray(ref, shape=shape, dtype=dtype)


@dispatcher.tensordot.register
@log_args(logger)
def tensordot(a: Sequence[COMPSsArray], b: Sequence[COMPSsArray], axes, method="sequential") -> COMPSsArray:
    dtype = np.result_type(a[0].dtype, b[0].dtype)
    shape = result_shape(a[0].shape, b[0].shape, axes)

    # TODO refactor method names
    if method == "sequential":
        a = [i.data for i in a]
        b = [i.data for i in b]
        ref = task.tensordot.sequential(a, b, axes)
    elif method == "commutative":
        ref = MaybeArray()
        for ia, ib in zip(a, b):
            task.tensordot.commutative(ref, ia, ib, axes)
    elif method == "commutative-but-first":
        ref = task.tensordot.tensordot(a[0].data, b[0].data, axes)
        for ia, ib in zip(a[1:], b[1:]):
            task.tensordot.commutative(ref, ia.data, ib.data, axes)
    else:
        raise ValueError("invalid method")
    return COMPSsArray(ref, shape=shape, dtype=dtype)


@dispatcher.linalg.svd.register
@log_args(logger)
def svd(a: COMPSsArray, full_matrices=True, compute_uv=True, hermitian=False) -> Union[Tuple[COMPSsArray, COMPSsArray, COMPSsArray], COMPSsArray]:
    assert a.ndim >= 2
    n = a.shape[-1]
    m = a.shape[-2]
    k = min(m, n)
    rest = a.shape[0:-2]

    if compute_uv:
        U, s, Vh = task.svd(a.data, full_matrices=full_matrices, hermitian=hermitian)

        # TODO check result dtype
        if full_matrices:
            U = COMPSsArray(U, shape=(*rest, m, m), dtype=a.dtype)
            s = COMPSsArray(s, shape=(*rest, k), dtype=a.dtype)
            Vh = COMPSsArray(Vh, shape=(*rest, n, n), dtype=a.dtype)
        else:
            U = COMPSsArray(U, shape=(*rest, m, k), dtype=a.dtype)
            s = COMPSsArray(s, shape=(*rest, k), dtype=a.dtype)
            Vh = COMPSsArray(Vh, shape=(*rest, k, n), dtype=a.dtype)

        return (U, s, Vh)

    else:
        s = task.svd_vals(a.data, hermitian=hermitian)
        s = COMPSsArray(s, shape=(*rest, k), dtype=a.dtype)

        return s


@dispatcher.linalg.qr.register
@log_args(logger)
def qr(a: COMPSsArray, mode="reduced"):
    n = a.shape[-1]
    m = a.shape[-2]
    k = min(m, n)
    rest = a.shape[0:-2]

    if mode == "complete":
        q, r = task.qr.qr_complete(a.data)
        q = COMPSsArray(q, shape=(*rest, m, m), dtype=a.dtype)
        r = COMPSsArray(r, shape=(*rest, m, n), dtype=a.dtype)
        return (q, r)

    elif mode == "reduced":
        q, r = task.qr.qr_reduced(a.data)
        q = COMPSsArray(q, shape=(*rest, m, k), dtype=a.dtype)
        r = COMPSsArray(r, shape=(*rest, k, n), dtype=a.dtype)
        return (q, r)

    elif mode == "r":
        r = task.qr.qr_r(a.data)
        r = COMPSsArray(r, shape=(*rest, k, n), dtype=a.dtype)
        return r

    elif mode == "raw":
        h, tau = task.qr_raw(a.data)
        h = COMPSsArray(h, shape=(*rest, n, m), dtype=a.dtype)
        tau = COMPSsArray(tau, shape=(*rest, k), dtype=a.dtype)
        return (h, tau)

    else:
        raise ValueError(f'mode must be one of "reduced", "complete", "r" or "raw" but is {mode}')


@dispatcher.cumsum.register
@log_args(logger)
def cumsum(a: COMPSsArray, axis=None, dtype=None, out=None):
    if out:
        assert isinstance(out, COMPSsArray)
        task.cumsum_out(out.data, a.data, axis=axis, dtype=dtype)
    else:
        ref = task.cumsum(a.data, axis=axis, dtype=dtype)
        shape = tuple(filter(lambda x: x[0] != axis, enumerate(a.shape)))
        dtype = dtype or a.dtype
        return COMPSsArray(ref, shape=shape, dtype=dtype)


@dispatcher.count_nonzero.register
@log_args(logger)
def count_nonzero(a: COMPSsArray, axis=None, keepdims=False) -> Union[int, COMPSsArray]:
    ref = task.count_nonzero(a.data, axis, keepdims)

    if axis is None:
        ret = compss_wait_on(ref)
        return ret
    else:
        shape = list(a.shape)
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        shape = tuple(shape)

        ret = COMPSsArray(ref, shape=shape, dtype=np.int0)
        return ret


@dispatcher.einsum.register
@log_args(logger)
def einsum(pattern: str, *operands: COMPSsArray, out: Optional[COMPSsArray] = None, dtype=None, order="K", casting="safe", optimize=False):
    if out is None:
        inputs, output, _ = parse_einsum_input((pattern, *operands))

        shape = find_output_shape(inputs, [op.shape for op in operands], output)
        dtype = np.result_type(*[op.dtype for op in operands])
        data = task.einsum(pattern, *operands, dtype=dtype, order=order, casting=casting, optimize=optimize)

        return COMPSsArray(data, shape=shape, dtype=dtype)

    else:
        task.einsum(pattern, *operands, out=out.data, dtype=dtype, order=order, casting=casting, optimize=optimize)
        return out


# @implements(np.block, COMPSsArray)
# def __compss_block(arrays):
#     return np.block(compss_wait_on([a.data for a in arrays]))


@log_args(logger)
def rand(shape):
    # TODO support inner as in BlockArray
    dtype = np.dtype(np.float64)
    return COMPSsArray(task.init.rand(shape), shape=shape, dtype=dtype)
