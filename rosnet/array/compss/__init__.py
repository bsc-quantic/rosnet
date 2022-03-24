import logging
from typing import Tuple, Sequence, Union
import functools
from copy import deepcopy
from math import prod
import numpy as np
from pycompss.runtime.management.classes import Future as COMPSsFuture
from pycompss.api.api import compss_delete_object, compss_wait_on
from rosnet.core.macros import todo
from rosnet.core.math import result_shape, isunique
from rosnet.core.interface import Array, ArrayConvertable, AsyncArray
from rosnet.core.mixin import ArrayFunctionMixin
from rosnet import tuning, dispatch as dispatcher
from . import task
from rosnet.array.block import BlockArray
from rosnet.array.maybe import MaybeArray

logger = logging.getLogger(__name__)

try:
    from rosnet.array.compss.dataclay import DataClayBlock
    from numpy.core import umath as um
    from numpy.lib.mixins import _binary_method, _numeric_methods, _reflected_binary_method, _unary_method

    DATACLAY = True

    # NOTE monkey-patch to implement np.lib.mixins.NDArrayOperatorsMixin for DataClayBlock. please contact Alex, Javi and Sergio for explanations.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self.__array_ufunc_no_expansion__(ufunc, method, inputs, kwargs)

    DataClayBlock.__array_ufunc__ = __array_ufunc__

    DataClayBlock.__lt__ = _binary_method(um.less, "lt")
    DataClayBlock.__le__ = _binary_method(um.less_equal, "le")
    DataClayBlock.__eq__ = _binary_method(um.equal, "eq")
    DataClayBlock.__ne__ = _binary_method(um.not_equal, "ne")
    DataClayBlock.__gt__ = _binary_method(um.greater, "gt")
    DataClayBlock.__ge__ = _binary_method(um.greater_equal, "ge")

    DataClayBlock.__add__, DataClayBlock.__radd__, DataClayBlock.__iadd__ = _numeric_methods(um.add, "add")
    DataClayBlock.__sub__, DataClayBlock.__rsub__, DataClayBlock.__isub__ = _numeric_methods(um.subtract, "sub")
    DataClayBlock.__mul__, DataClayBlock.__rmul__, DataClayBlock.__imul__ = _numeric_methods(um.multiply, "mul")
    DataClayBlock.__matmul__, DataClayBlock.__rmatmul__, DataClayBlock.__imatmul__ = _numeric_methods(um.matmul, "matmul")
    DataClayBlock.__truediv__, DataClayBlock.__rtruediv__, DataClayBlock.__itruediv__ = _numeric_methods(um.true_divide, "truediv")
    DataClayBlock.__floordiv__, DataClayBlock.__rfloordiv__, DataClayBlock.__ifloordiv__ = _numeric_methods(um.floor_divide, "floordiv")
    DataClayBlock.__mod__, DataClayBlock.__rmod__, DataClayBlock.__imod__ = _numeric_methods(um.remainder, "mod")
    DataClayBlock.__divmod__ = _binary_method(um.divmod, "divmod")
    DataClayBlock.__rdivmod__ = _reflected_binary_method(um.divmod, "divmod")
    DataClayBlock.__pow__, DataClayBlock.__rpow__, DataClayBlock.__ipow__ = _numeric_methods(um.power, "pow")
    DataClayBlock.__lshift__, DataClayBlock.__rlshift__, DataClayBlock.__ilshift__ = _numeric_methods(um.left_shift, "lshift")
    DataClayBlock.__rshift__, DataClayBlock.__rrshift__, DataClayBlock.__irshift__ = _numeric_methods(um.right_shift, "rshift")
    DataClayBlock.__and__, DataClayBlock.__rand__, DataClayBlock.__iand__ = _numeric_methods(um.bitwise_and, "and")
    DataClayBlock.__xor__, DataClayBlock.__rxor__, DataClayBlock.__ixor__ = _numeric_methods(um.bitwise_xor, "xor")
    DataClayBlock.__or__, DataClayBlock.__ror__, DataClayBlock.__ior__ = _numeric_methods(um.bitwise_or, "or")

    DataClayBlock.__neg__ = _unary_method(um.negative, "neg")
    DataClayBlock.__pos__ = _unary_method(um.positive, "pos")
    DataClayBlock.__abs__ = _unary_method(um.absolute, "abs")
    DataClayBlock.__invert__ = _unary_method(um.invert, "invert")

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
        logger.debug(f"__del__: id={id(self)}, self={self}")
        if isinstance(self.data, COMPSsFuture):
            compss_delete_object(self.data)
        elif DATACLAY:
            if isinstance(self.data, DataClayBlock):
                self.data.session_detach()  # TODO is this call ok?

    def __str__(self) -> str:
        return f"COMPSsArray<data=id({id(self.data)}), shape={self.shape}, dtype={self.dtype}>"

    def __repr__(self) -> str:
        return f"COMPSsArray<data=id({id(self.data)}), shape={self.shape}, dtype={self.dtype}>"

    def __getitem__(self, idx) -> COMPSsFuture:
        logger.debug(f"__getitem__: self={self}, shape={self.shape}, idx={idx}")
        return compss_wait_on(task.getitem(self.data, idx))

    def __setitem__(self, key, value):
        logger.debug(f"__setitem__: self={self}, key={key}, value={value}")
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

    def __deepcopy__(self, memo):
        if isinstance(self.data, COMPSsFuture):
            ref = task.copy(self.data)
        elif DATACLAY:
            if isinstance(self.data, DataClayBlock):
                ref = self.data.dc_clone()
        else:
            ref = deepcopy(self.data)
        return COMPSsArray(ref, shape=self.shape, dtype=self.dtype)

    def __array__(self) -> np.ndarray:
        return dispatcher.to_numpy(self.data)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if ufunc.nin > 2:
            return NotImplemented

        # get COMPSs reference if COMPSsArray
        inputs_unwrap = [arg.data if isinstance(arg, AsyncArray) else arg for arg in inputs]

        logger.debug(f"__array_ufunc__:\n\tufunc={ufunc},\n\tmethod={method},\n\tinputs={inputs},\n\tkwargs={kwargs}")

        # TODO fix
        inplace = False
        if "out" in kwargs and kwargs["out"] == (self,):
            inplace = True
            kwargs["out"] = (self.data,)

        # 'at' operates in-place
        if method == "at":
            if not np.can_cast(inputs[1], inputs[0], casting="safe"):
                return NotImplemented
            task.ioperate(ufunc, *inputs_unwrap, **kwargs)

        # '__call__', 'outer'
        elif method in "__call__":
            if inplace:
                types = [i.dtype if hasattr(i, "dtype") else i for i in inputs]
                if not np.can_cast(types[1], types[0], casting="safe"):
                    return NotImplemented
                task.ioperate(ufunc, *inputs_unwrap, **kwargs)
                return self
            else:
                ref = task.operate(ufunc, *inputs_unwrap, **kwargs)
                dtype = np.result_type(*(i.dtype if hasattr(i, "dtype") else i for i in inputs))

                logger.debug(f"ref={ref}, dtype={dtype}")
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


# COMPSsArray is an async array
AsyncArray.register(COMPSsArray)


@dispatcher.to_numpy.register
def _(arr: COMPSsFuture):
    return compss_wait_on(arr)


@dispatcher.to_numpy.register
def to_numpy(arr: COMPSsArray):
    return dispatcher.to_numpy(arr.data)


@dispatcher.to_numpy.register
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


def zeros(shape, dtype=None, order="C") -> COMPSsArray:
    return full(shape, 0, dtype=dtype, order=order)


def ones(shape, dtype=None, order="C") -> COMPSsArray:
    return full(shape, 1, dtype=dtype, order=order)


def full(shape, fill_value, dtype=None, order="C") -> COMPSsArray:
    ref = task.full(shape, fill_value, dtype=dtype, order=order)
    return COMPSsArray(ref, shape=shape, dtype=dtype or np.dtype(type(fill_value)))


@dispatcher.zeros_like.register
def zeros_like(a: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> Union[np.ndarray, COMPSsArray]:
    if subok:
        return zeros(shape or a.shape, dtype=dtype or a.dtype, order=order)
    else:
        return np.zeros(shape or a.shape, dtype=dtype or a.dtype, order=order)


@dispatcher.ones_like.register
def ones_like(a: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> Union[np.ndarray, COMPSsArray]:
    if subok:
        return ones(shape or a.shape, dtype=dtype or a.dtype, order=order)
    else:
        return np.ones(shape or a.shape, dtype=dtype or a.dtype, order=order)


@dispatcher.full_like.register
def full_like(a: COMPSsArray, fill_value, dtype=None, order="K", subok=True, shape=None) -> Union[np.ndarray, COMPSsArray]:
    if subok:
        return full(shape or a.shape, fill_value, dtype=dtype or a.dtype, order=order)
    else:
        return np.full(shape or a.shape, fill_value, dtype=dtype or a.dtype, order=order)


@dispatcher.empty_like.register
def empty_like(prototype: COMPSsArray, dtype=None, order="K", subok=True, shape=None) -> COMPSsArray:
    pass


@dispatcher.reshape.register
def reshape(a: COMPSsArray, shape, order="C", inplace=False):
    logger.debug(f"reshape: a={a}, shape={shape}, order={order}, inplace={inplace}")
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
def transpose(a: COMPSsArray, axes=None, inplace=False):
    logger.debug(f"transpose: a={a}, axes={axes}, inplace={inplace}")
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
def stack(arrays: Sequence[COMPSsArray], axis=0, out=None) -> COMPSsArray:
    pass


@todo
@dispatcher.split.register
def split(array: COMPSsArray, indices_or_sections, axis=0) -> Sequence[COMPSsArray]:
    pass


@dispatcher.tensordot.register(COMPSsArray, ArrayConvertable)
@dispatcher.tensordot.register(ArrayConvertable, COMPSsArray)
def tensordot(a: Union[COMPSsArray, ArrayConvertable], b: Union[COMPSsArray, ArrayConvertable], axes):
    a = a if isinstance(a, COMPSsArray) else COMPSsArray(a)
    b = b if isinstance(b, COMPSsArray) else COMPSsArray(b)
    return dispatcher.tensordot[(COMPSsArray, COMPSsArray)](a, b, axes)


@dispatcher.tensordot.register
def tensordot(a: COMPSsArray, b: COMPSsArray, axes) -> COMPSsArray:
    dtype = np.result_type(a.dtype, b.dtype)
    shape = result_shape(a.shape, b.shape, axes)

    ref = task.tensordot.tensordot(a.data, b.data, axes)
    return COMPSsArray(ref, shape=shape, dtype=dtype)


@dispatcher.tensordot.register
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
def count_nonzero(a: COMPSsArray, axis=None, keepdims=False) -> Union[int, COMPSsArray]:
    ref = task.count_nonzero(a.data, axis, keepdims)

    if axis is None:
        ret = compss_wait_on(ref)
        logger.debug(f"count_nonzero: a={a}, axis={axis}, keepdims={keepdims}, result={ret}")
        return ret
    else:
        shape = list(a.shape)
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        shape = tuple(shape)

        ret = COMPSsArray(ref, shape=shape, dtype=np.int0)
        logger.debug(f"count_nonzero: a={a}, axis={axis}, keepdims={keepdims}, result={ret}")
        return ret


# @implements(np.block, COMPSsArray)
# def __compss_block(arrays):
#     return np.block(compss_wait_on([a.data for a in arrays]))


def rand(shape):
    # TODO support inner as in BlockArray
    dtype = np.dtype(np.float64)
    return COMPSsArray(task.init.rand(shape), shape=shape, dtype=dtype)
