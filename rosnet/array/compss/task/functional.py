import functools
from pycompss.api.parameter import IN, INOUT
from rosnet.tuning.task import autotune


# TODO `reduce` when more arguments?
# TODO change __name__ for str(func)
@autotune(returns=1)
def operate(func, *args, **kwargs):
    "Apply operator."
    return func(*args, **kwargs)


# TODO `reduce` when more arguments?
# TODO change __name__ for str(func)
@autotune(a=INOUT, returns=1)
def ioperate(func, a, *args, **kwargs):
    "Apply in-place operator."
    func(a, *args, **kwargs)


@autotune(out=INOUT, returns=0)
def ufunc_out(out, ufunc, *args, **kwargs):
    ufunc(*args, out=out, **kwargs)
