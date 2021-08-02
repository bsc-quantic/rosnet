from typing import Tuple, Type, List
import asyncio
import copy
import functools
import itertools
import operator as op
import numpy as np
from pycompss.api.api import compss_delete_object, compss_wait_on
from rosnet import task, utils
from rosnet.utils import todo, isunique, ndarray_from_list, prod, space
from rosnet.helper import implements, numpy_dispatcher


def assert_axes(axes):
    assert isinstance(axes, (list, tuple)) and all(isinstance(ax, (list, tuple)) for ax in axes)
    assert all(isunique(ax) for ax in axes)
    assert len(axes[0]) == len(axes[1])


# TODO special variation for in-place functions? keep np.reshape/transpose/... as non-modifying -> create new COMPSsArray/BlockArray
# TODO support more properties of ndarray
class COMPSsArray(np.lib.mixins.NDArrayOperatorsMixin):
    """ Reference to a `numpy.ndarray` managed by COMPSs.

    Unlike a `numpy.ndarray`, a `COMPSsArray` is mutable and does not return views. As such, the following methods act in-place and return nothing:
    - `reshape`
    - `transpose`
    """
    # pylint: disable=protected-access

    def __init__(self, *args, **kwargs):
        self._ref = args[0]
        if issubclass(args[0].__class__, np.ndarray) or isinstance(args[0], np.ndarray):
            self.__shape = args[0].shape
            self.__dtype = args[0].dtype
        elif asyncio.isfuture(args[0]):
            self.__shape = kwargs['shape']
            self.__dtype = kwargs['dtype']
        else:
            raise TypeError(
                f"You must provide a np.ndarray or a COMPSs Future to a np.ndarray, but a {type(args[0])} was provided")

    def __del__(self):
        compss_delete_object(self._ref)

    def __getitem__(self, idx):
        return compss_wait_on(task.getitem(self._ref, idx))

    def __setitem__(self, key, value):
        task.setitem(self._ref, key, value)

    @property
    def shape(self) -> Tuple[int]:
        return self.__shape

    @todo
    @shape.setter
    def _(self, value: Tuple[int]):
        # self.reshape(value)
        pass

    @property
    def size(self) -> int:
        return utils.prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.__shape)

    @property
    def dtype(self) -> Type:
        return self.__dtype

    def __deepcopy__(self, memo):
        ref = task.copy(self._ref)
        return COMPSsArray(ref, shape=self.shape, dtype=self.dtype)

    # def sync(self):
    #     self._ref = compss_wait_on(self._ref)

    # numpy-compatibility
    def __array__(self) -> np.ndarray:
        return compss_wait_on(self._ref)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if ufunc.nin > 2:
            return NotImplemented

        # get COMPSs reference if COMPSsArray
        inputs = [arg._ref if isinstance(arg, self.__class__) else arg
                  for arg in inputs]

        inplace = False
        if 'out' in kwargs and kwargs['out'] == (self,):
            inplace = True
            kwargs['out'] = (self._ref,)

        # 'at' operates in-place
        if method == 'at':
            if not np.can_cast(inputs[1], inputs[0], casting='safe'):
                return NotImplemented
            task.ioperate(ufunc, *inputs, **kwargs)

        # '__call__', 'outer'
        elif method in '__call__':
            if inplace:
                types = [i.dtype if hasattr(i, 'dtype') else i for i in inputs]
                if not np.can_cast(types[1], types[0], casting='safe'):
                    return NotImplemented
                task.ioperate(ufunc, *inputs, **kwargs)
                return self
            else:
                ref = task.operate(ufunc, *inputs, **kwargs)
                dtype = np.result_type(*(i.dtype if hasattr(i, 'dtype') else i for i in inputs))
                return COMPSsArray(ref, shape=self.shape, dtype=dtype)

        elif method == 'outer':
            if inplace:
                return NotImplemented
            else:
                ref = task.operate(ufunc, *inputs, **kwargs)
                shape = functools.reduce(tuple.__add__, (i.shape for i in inputs))
                dtype = np.result_type(*(i.dtype for i in inputs))
                return COMPSsArray(ref, shape=shape, dtype=dtype)

        # 'reduce', 'accumulate', 'reduceat' not supported yet
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in numpy_dispatcher[self.__class__]:
            print('not in dispatcher')
            return NotImplemented
        if not all(t == self.__class__ for t in types):
            print('bad type')
            # TODO create COMPSsArray if np.ndarray implementation
            return NotImplemented

        return numpy_dispatcher[self.__class__][func](*args, **kwargs)

# numpy: chainging array shape
@implements(np.reshape, COMPSsArray)
def __compss_reshape(a: COMPSsArray, newshape: Tuple[int], order='F'):
    # pylint: disable=protected-access
    # TODO support order?
    # TODO should return something?
    task.reshape(a._ref, newshape)

# numpy: transpose-like operations
@implements(np.transpose, COMPSsArray)
def __compss_transpose(a: COMPSsArray, axes: Tuple[int] = None):
    # pylint: disable=protected-access
    # TODO should return something?
    task.transpose(a._ref, axes)

# numpy: joining arrays
@todo
@implements(np.stack, COMPSsArray)
def __compss_stack(self):
    pass

# numpy: splitting arrays
@todo
@implements(np.split, COMPSsArray)
def __compss_split(self, indices_or_sections, axis=0):
    pass

@implements(np.tensordot, COMPSsArray)
def __compss_tensordot(a: COMPSsArray, b: COMPSsArray, axes):
    # pylint: disable=protected-access
    # TODO assertions

    # only support operating against COMPSsArray
    if not all(isinstance(i, COMPSsArray) for i in (a,b)):
        return NotImplemented

    dtype = np.result_type(a.dtype, b.dtype)
    shape = result_shape(a.shape, b.shape, axes)

    ref = task.tensordot.tensordot(a._ref, b._ref, axes)
    return COMPSsArray(ref, shape=shape, dtype=dtype)


def __compss_tensordot_commutative(a: COMPSsArray, b: COMPSsArray, axes, buffer: COMPSsArray):
    # pylint: disable=protected-access
    task.tensordot.commutative(buffer._ref, a._ref, b._ref, axes)

# @implements(np.block, COMPSsArray)
# def __compss_block(arrays):
#     return np.block(compss_wait_on([a._ref for a in arrays]))


class BlockArray(np.lib.mixins.NDArrayOperatorsMixin):
    """ Block Array of `numpy.ndarray` or `COMPSsArray`.

    Assumptions:
    - All blocks have same `blockshape` and `dtype`.
    """

    _grid: np.ndarray = None
    __blockshape: Tuple = None
    __dtype: Type = None

    def __init__(self, *args, **kwargs):
        # TODO asserts
        if isinstance(args[0], COMPSsArray):
            self._grid = np.array(args[0])
            self.__blockshape = args[0].shape
            self.__dtype = args[0].dtype
        elif isinstance(args[0], np.ndarray):
            # assert asyncio.isfuture(args[0][0])
            self._grid = args[0].copy()
            self.__blockshape = kwargs['blockshape']
            self.__dtype = kwargs.get('dtype')
        else:
            raise TypeError("expected COMPSsArray or numpy.ndarray, got " % type(args[0]))

    def __str__(self):
        return "BlockArray(shape=%r, grid=%r, blockshape=%r)" % (self.shape, self.grid, self.blockshape)

    @todo
    def __getitem__(self, index):
        pass

    @todo
    def __setitem__(self, key, value):
        pass

    @property
    def shape(self) -> Tuple[int]:
        return tuple(np.multiply(self._grid.shape, self.__blockshape))

    @property
    def blockshape(self) -> Tuple[int]:
        return self.__blockshape

    @property
    def nblock(self) -> int:
        return self._grid.size

    @property
    def grid(self) -> Tuple[int]:
        return self._grid.shape

    @property
    def size(self) -> int:
        return utils.prod(self.shape)

    @property
    def blocksize(self) -> int:
        return utils.prod(self.blockshape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> Type:
        return self.__dtype

    def __deepcopy__(self, memo):
        grid = np.empty_like(self._grid)
        for i in self.nblock:
            grid.flat[i] = copy.deepcopy(self._grid.flat[i])

        return BlockArray(grid, self.blockshape)

    # numpy-compatibility
    def __array__(self) -> np.ndarray:
        blocks = np.empty_like(self._grid, dtype=object)
        it = np.nditer(self._grid, flags=['refs_ok', 'multi_index'], op_flags=['readonly'], op_axes=[tuple(range(self.ndim))])
        with it:
            for x in it:
                blocks[it.multi_index] = compss_wait_on(x[()]._ref)
        return np.block(blocks)

    @todo
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        pass

    def __array_function__(self, func, types, args, kwargs):
        if func not in numpy_dispatcher[self.__class__]:
            print('not in dispatcher')
            return NotImplemented
        if not all(t == self.__class__ for t in types):
            print('bad type')
            # TODO create COMPSsArray if np.ndarray implementation
            return NotImplemented

        return numpy_dispatcher[self.__class__][func](*args, **kwargs)

@todo
@implements(np.reshape, BlockArray)
def __block_reshape(a, shape):
    pass

@implements(np.transpose, BlockArray)
def __block_transpose(a, axes=None):
    # pylint: disable=protected-access
    if not isunique(axes):
        raise ValueError("'axes' must be a unique list: %s" % axes)

    # transpose blocks
    for ref in a._grid.flat:
        task.transpose(ref, axes)

    # transpose grid
    a._grid = np.transpose(a._grid, axes)

    # transpose block shape
    a.__blockshape = tuple(a.__blockshape[i] for i in axes)

@implements(np.tensordot, BlockArray)
def __block_tensordot(a: BlockArray, b: BlockArray, axes):
    # pylint: disable=protected-access
    # TODO assertions

    # only support operating against BlockArray
    if not isinstance(b, a.__class__):
        return NotImplemented

    # iterators
    outer_axes = [list(set(range(i.ndim)) - set(ax)) for ax, i in zip(axes, (a, b))]
    outer_iter_a, inner_iter_a = np.nested_iters(a._grid,
        [outer_axes[0], axes[0]],
        op_flags=['readonly'],
        flags=['multi_index', 'refs_ok'])
    outer_iter_b, inner_iter_b = np.nested_iters(b._grid,
        [outer_axes[1], axes[1]],
        op_flags=['readonly'],
        flags=['multi_index', 'refs_ok'])

    grid = np.empty(outer_iter_a.shape + outer_iter_b.shape, dtype=COMPSsArray)
    dtype = np.result_type(a.dtype, b.dtype)
    blockshape = result_shape(a.blockshape, b.blockshape, axes)

    # TODO dynamic parallelism
    for outer_i_a in outer_iter_a:
        for outer_i_b in outer_iter_b:
            buffer = COMPSsArray(task.init.full(blockshape, 0, dtype), shape=blockshape, dtype=dtype)

            for inner_i_a, inner_i_b in zip(inner_iter_a, inner_iter_b):
                __compss_tensordot_commutative(inner_i_a[()], inner_i_b[()], axes, buffer)

            grid[outer_iter_a.multi_index + outer_iter_b.multi_index] = buffer
            inner_iter_a.reset()
            inner_iter_b.reset()
        outer_iter_b.reset()

    return BlockArray(grid, blockshape=blockshape, dtype=dtype)


def result_nblock(a: BlockArray, b: BlockArray, axes):
    " Returns the number of blocks of the resulting array after `tensordot`. "
    return prod(prod(itertools.compress(grid, [x not in ax for x in range(len(grid))])) for ax, grid in zip(axes, (a.grid, b.grid)))

def result_shape(a, b, axes):
    " Returns the blockshape of the resulting array after `tensordot`. "
    outer_axes = [tuple(set(range(len(bs))) - set(ax)) for ax, bs in zip(axes, (a, b))]
    return functools.reduce(op.add, (tuple(i[ax] for ax in outer_ax) for outer_ax, i in zip(outer_axes, (a,b))))


def array(arr: np.ndarray, blockshape=None):
    shape = arr.shape
    blockshape = shape if blockshape is None else blockshape
    grid = [s // bs for s, bs in zip(shape, blockshape)]

    blocks = []
    for bidx in space(grid):
        idx_begin = [x*y for x, y in zip(blockshape, bidx)]
        idx_end = [x+y for x, y in zip(blockshape, idx_begin)]
        idx = tuple(slice(b, e) for b, e in zip(idx_begin, idx_end))
        block = arr[idx]
        blocks.append(COMPSsArray(block, blockshape=blockshape))

    blocks = ndarray_from_list(blocks, grid)
    return BlockArray(blocks, blockshape=blockshape)


def zeros(shape, blockshape=None, dtype=None):
    return full(0, shape, blockshape, dtype)


def ones(shape, blockshape=None, dtype=None):
    return full(1, shape, blockshape, dtype)


def full(value, shape, blockshape=None, dtype=None):
    blockshape = shape if blockshape is None else blockshape
    grid = tuple(s // bs for s, bs in zip(shape, blockshape))

    blocks = [task.full(blockshape, value, dtype) for _ in range(prod(grid))]

    blocks = ndarray_from_list(blocks, grid)
    return BlockArray(blocks, blockshape)
