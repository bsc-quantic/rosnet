from itertools import chain

import autoray
import numpy as np

try:
    from .dataclay import DataClayBlock

    def do(fn_name: str, *args, **kwargs):
        """Wrapper for 'autoray.do' when DataClayBlocks are involved.
        In order to execute in dataClay memory space, a DataClayObject must be called from a method. This wrapper does so by detecting if any of the arguments is a DataClayBlock and calling the '__array_ufunc_no_expansion__'  and '__array_function__' methods, with the function returned by 'autoray.get_lib_fn'.
        """
        backend = kwargs.pop("like", "numpy")
        fn = autoray.get_lib_fn(backend, fn_name)

        out = kwargs.get("out", None)
        if any(isinstance(arg, DataClayBlock) for arg in chain(args, [out])):
            caller: DataClayBlock = out or next(isinstance(arg, DataClayBlock) for arg in args)
            if isinstance(fn, np.ufunc):
                # NOTE "method" arg not supported
                return caller.__array_ufunc_no_expansion__(fn, "", args, kwargs)
            else:
                # NOTE "types" arg not supported
                return caller.__array_function__(fn, ..., args, kwargs)

        return fn(*args, **kwargs)

except ImportError:
    do = autoray.do
