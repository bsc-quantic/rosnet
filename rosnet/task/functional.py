import functools
from pycompss.api.task import task
from pycompss.api.parameter import IN, INOUT
from rosnet.utils import todo


# TODO generic_task
# TODO `reduce` when more arguments?
# TODO change __name__ for str(func)
@task(returns=1)
def operate(func, *args, **kwargs):
    " Apply operator. "
    return func(*args, **kwargs)


# TODO generic_task
# TODO `reduce` when more arguments?
# TODO change __name__ for str(func)
@task(a=INOUT, returns=1, label='func')
def ioperate(func, a, *args, **kwargs):
    " Apply in-place operator. "
    func(a, *args, **kwargs)