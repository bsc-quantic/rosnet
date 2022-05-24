from numpy.core import umath as um
from numpy.lib.mixins import _binary_method, _numeric_methods, _reflected_binary_method, _unary_method

# NOTE If launch from COMPSs, this should already managed by them
from dataclay.api import init

init()

from rosnet_dclaymodel.model import DataClayBlock


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
