# Ecosystem integration

One of the goals of `rosnet` is to be compatible with the Python ecosystem, so that users find `rosnet` as easy to use as possible. Below you can find a list of libraries that we explicitly support and test for interoperability.

## NumPy

Array classes defined in `rosnet` support the NumPy dispatch-mechanism  by implementing the `__array__` , `__array_ufunc__` ([NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html)), `__array_function__` ([NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html)) dundler methods. Thus, any dispatchable NumPy method should work on `rosnet`-defined array classes.

```{note}
This is work in progress. Not all methods are supported yet but you should see more implementations available as development continues. Status is tracked at [#31](https://github.com/UB-Quantic/rosnet/issues/31).
```

## AutoRAY

`autoray` is an array method dispatcher that powers `quimb`. It offers a common interface among diferent array backends like `cupy`, `jax`, `tensorflow` and many more. By wrapping functions,
it extends or modifies the API. `rosnet` is wraps some functions to offer the API expected by `autoray` without any breakage with other dependencies.
