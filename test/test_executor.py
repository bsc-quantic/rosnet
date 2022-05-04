import inspect
import operator

import numpy as np
import pytest
from rosnet.executor import modulepath, task_directions

pycompss = pytest.importorskip("pycompss")
from pycompss.api.parameter import INOUT

builtin_operators = [attr for attr in operator.__dict__.values() if inspect.isbuiltin(attr)]
builtin_operators_inplace = [
    operator.iadd,
    operator.iand,
    operator.iconcat,
    operator.ifloordiv,
    operator.ilshift,
    operator.imod,
    operator.imul,
    operator.imatmul,
    operator.ior,
    operator.ipow,
    operator.irshift,
    operator.isub,
    operator.itruediv,
    operator.ixor,
]


class TestModulePathBuiltins:
    @pytest.mark.parametrize("builtin", filter(inspect.isbuiltin, operator.__dict__.values()))
    def test_operators(self, builtin):
        assert modulepath(builtin) == ["builtins", builtin.__name__]


class TestTaskDirectionsBuiltins:
    def test_setitem(self):
        assert task_directions(operator.setitem) == {"returns": 0, "a": INOUT}

    @pytest.mark.parametrize("builtin", filter(lambda x: x not in builtin_operators_inplace and x is not operator.setitem, builtin_operators))
    def test_builtins(self, builtin):
        assert task_directions(builtin) == {"returns": 1}

    @pytest.mark.parametrize("builtin", builtin_operators_inplace)
    def test_builtins_inplace(self, builtin):
        assert task_directions(builtin) == {"returns": 1, "a": INOUT}


ufuncs = [attr for attr in np.__dict__.values() if isinstance(attr, np.ufunc)]
np_functions = [attr for attr in np.__dict__.values() if inspect.isfunction(attr)]
np_functions_aliased = [
    np.show_config,
    np.set_string_function,
    np.disp,
    np.add_newdoc,
    np.get_array_wrap,
    np.deprecate,
    np.deprecate_with_doc,
    np.get_include,
    np.who,
    np.byte_bounds,
    np.safe_eval,
    np.recfromtxt,
    np.recfromcsv,
]
np_linalg_functions = [attr for attr in np.linalg.__dict__.values() if inspect.isfunction(attr)]


class TestModulePathNumpy:
    @pytest.mark.parametrize("ufunc", ufuncs)
    def test_ufunc(self, ufunc):
        assert modulepath(ufunc) == ["numpy", "ufunc"]

    @pytest.mark.parametrize("ufunc", ufuncs)
    @pytest.mark.parametrize("method", ["reduce", "accumulate", "reduceat", "outer", "at"])
    def test_ufunc_methods(self, ufunc, method):
        assert modulepath(getattr(ufunc, method)) == ["numpy", "ufunc", method]

    @pytest.mark.parametrize("function", [func if func not in np_functions_aliased else pytest.param(func, marks=pytest.mark.xfail(reason="Function is aliased")) for func in np_functions])
    def test_numpy_functions(self, function):
        assert modulepath(function) == ["numpy", function.__name__]

    @pytest.mark.parametrize("function", np_linalg_functions)
    def test_numpy_linalg_functions(self, function):
        assert modulepath(function) == ["numpy", "linalg", function.__name__]


class TestTaskDirectionsNumpy:
    def test_copyto(self):
        assert task_directions(np.copyto) == {"returns": 0, "dst": INOUT}

    @pytest.mark.parametrize("ufunc", ufuncs)
    def test_ufunc(self, ufunc):
        assert task_directions(ufunc) == {"returns": 1}
        assert ufunc.nout == 1

    @pytest.mark.parametrize("ufunc", ufuncs)
    @pytest.mark.parametrize(
        "method",
        [
            "reduce",
            "accumulate",
            "outer",
            pytest.param("reduceat", marks=pytest.mark.xfail),
            pytest.param("at", marks=pytest.mark.xfail),
        ],
    )
    def test_ufunc_methods(self, ufunc, method):
        assert task_directions(getattr(ufunc, method)) == {"returns": 1}
        assert ufunc.nout == 1
