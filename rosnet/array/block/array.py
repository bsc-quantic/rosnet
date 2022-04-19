import sys
from typing import TypeVar, Generic, Optional, Sequence
from math import prod
from copy import deepcopy

if sys.version_info >= (3, 9):
    from types import GenericAlias
else:
    from typing_extensions import GenericAlias

import numpy as np

from rosnet import dispatch
from rosnet.core.abc import Array, ArrayConvertable
from rosnet.core.macros import todo
from rosnet.core.mixin import ArrayFunctionMixin
from rosnet.core.util import isunique, join_idx, measure_shape, nest_level, result_shape, space

T = TypeVar("T")


class BlockArray(Generic[T], np.lib.mixins.NDArrayOperatorsMixin, ArrayFunctionMixin):
    """A n-dimensional array divided in blocks.

    Implementation notes
    --------------------
    - All blocks are expected to have the same type and `dtype`.
    - All blocks are expected to be equally sized.
    - Automatic parametric type detection works only on Python 3.9 or later. On earlier versions, you must
    """

    data: np.ndarray = None  # type: ignore

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], list):
            self.__init_with_list__(*args, **kwargs)
        elif isinstance(args[0], ArrayConvertable):
            self.__init_with_array__(*args, **kwargs)
        else:
            raise ValueError("invalid constructor")

        self.__orig_class__ = GenericAlias(self.__class__, self.data.flat[0].__class__)

    def __init_with_list__(self, blocks: list, grid: Optional[Sequence[int]] = None):
        """Constructor.

        Arguments
        ---------
        - blocks: list[Array]. Nested list of arrays.
        """
        if grid is None:
            grid = measure_shape(blocks)

        self.data = np.empty(grid, dtype=object)

        it = np.nditer(self.data, flags=["refs_ok", "multi_index"], op_flags=["writeonly"])

        with it:
            for block in it:
                # case for nested list of arrays
                if isinstance(blocks[0], list):
                    data = blocks
                    for i in it.multi_index:
                        data = data[i]
                # case for list of arrays
                else:
                    data = blocks[it.iterindex]

                if data.ndim != self.data.ndim:
                    raise ValueError("blocks and grid should have same ndim. append single-dimensions (1) where needed")

                if isinstance(data, Array):
                    block[()] = data
                elif isinstance(data, ArrayConvertable):
                    block[()] = np.array(data)
                else:
                    raise ValueError("blocks must provide an array-like interface")

    def __init_with_array__(self, arr):
        """Constructor."""

        if nest_level(arr):
            self.data = arr
        else:
            self.data = np.empty(tuple(1 for _ in arr.shape), dtype=object)
            self.data.flat[0] = arr

    def __class_getitem__(cls, item):
        if not issubclass(item, Array):
            raise TypeError(f"'BlockArray[{item.__name__}]' is not a valid type as '{item.__name__}' does not fulfill the Array protocol")

        return super().__class_getitem__(item)

    @classmethod
    def __infer_type_parameter__(cls, *args, **kwargs) -> type:
        "Returns the type parameter."

        x = args[0]
        try:
            while True:
                x = x.flat[0] if isinstance(x, np.ndarray) else x[0]
        except:
            return type(x)

    def __str__(self):
        return "BlockArray(shape=%r, grid=%r, blockshape=%r, dtype=%r)" % (
            self.shape,
            self.grid,
            self.blockshape,
            self.dtype,
        )

    def __getitem__(self, index: Sequence[int]):
        # TODO advanced indexing
        if len(index) != self.ndim:
            raise IndexError(f"Invalid indexing: index={index}")

        gid = [i // s for i, s in zip(index, self.blockshape)]
        bid = [i % s for i, s in zip(index, self.blockshape)]

        return self.data[gid][bid]

    def __setitem__(self, key: Sequence[int], value):
        # TODO advanced indexing
        if len(key) != self.ndim:
            raise IndexError(f"Invalid indexing: key={key}")

        gid = [i // s for i, s in zip(key, self.blockshape)]
        bid = [i % s for i, s in zip(key, self.blockshape)]

        self.data[gid][bid] = value

    @property
    def shape(self) -> "tuple[int, ...]":
        return tuple(np.multiply(self.grid, self.blockshape))

    @property
    def blockshape(self) -> "tuple[int, ...]":
        return self.data.flat[0].shape

    @property
    def nblock(self) -> int:
        return self.data.size

    @property
    def grid(self) -> "tuple[int, ...]":
        return self.data.shape

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
    def blocksize(self) -> int:
        return prod(self.blockshape)

    @property
    def blocknbytes(self) -> int:
        return self.blocksize * self.itemsize

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.data.flat[0].dtype

    def __deepcopy__(self, memo):
        grid = np.empty_like(self.data)
        for i in range(self.nblock):
            grid.flat[i] = deepcopy(self.data.flat[i])

        return BlockArray(grid)

    def __array__(self) -> np.ndarray:
        "Returns a numpy.ndarray. Uses class-parametric specialization with multimethod."
        return dispatch.to_numpy(self)

    @todo
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        pass


__all__ = ["BlockArray"]
