from dataclay import DataClayObject, dclayMethod


class DataClayBlock(DataClayObject):
    """A persistent block class.

    @dclayImport numpy as np
    @ClassField data numpy.ndarray
    """

    @dclayMethod(arr="numpy.ndarray")
    def __init__(self, arr):
        self.data = arr

    @dclayMethod(key="anything", return_="anything")
    def __getitem__(self, key):
        return self.data[key]

    @dclayMethod(key="anything", value="anything")
    def __setitem__(self, key, value):
        if key == ():
            self.data = value
        self.data[key] = value

    @dclayMethod(return_="numpy.ndarray")
    def __array__(self):
        return self.data

    @dclayMethod(ufunc="numpy.ufunc", method="str", inputs="list", kwargs="dict", return_="anything")
    def __array_ufunc_no_expansion__(self, ufunc, method, inputs, kwargs):
        """Bypasses computation to dataClay memory space. Uses numpy dispatch mechanism to call the correct implementation.

        NOTE
        ----
        dataClay does not support argument expansion like *args, **kwargs.
        """

        # translate DataClayBlock to numpy.ndarray
        inputs = [i.data if isinstance(i, type(self)) else i for i in inputs]

        # inplace case
        out = kwargs.get("out", None)
        if isinstance(out, self.__class__):
            # NOTE contact Alex Barcelo if crash
            print("Alex was here and suspected something...")
            kwargs["out"] = kwargs["out"].data

        # call numpy function
        result = ufunc(*inputs, **kwargs)

        if isinstance(out, self.__class__):
            return out

        result = DataClayBlock(result)
        result.make_persistent()
        return result

    # TODO use autoray for calling and selecting backend
    @dclayMethod(function="anything", types="list", inputs="list", kwargs="dict", return_="anything")
    def __array_function__(self, function, types, inputs, kwargs):
        "Bypasses computation to dataClay memory space. Uses numpy dispatch mechanism to call the correct implementation."

        # translate DataClayBlock to numpy.ndarray
        inputs = [i.data if isinstance(i, type(self)) else i for i in inputs]

        # call numpy function
        result = function(*inputs, **kwargs)

        # wrap result into DataClayBlock
        result = DataClayBlock(result)
        result.make_persistent()
        return result
