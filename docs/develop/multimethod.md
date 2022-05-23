# Motivation for multiple-dispatch

During the development of `rosnet`, it was made clear that **multimethods** (aka multiple-dispatch, free functions, ...) eased the project complexity and facilitated a more modular organization.

Let's see an example with the `tensordot` implementation for `BlockArray` and `COMPSsArray`.


```python
class BlockArray:
	...

	def tensordot(self, other: "BlockArray", axes) -> "BlockArray":
		...
```

```python
class COMPSsArray:
	...

	def tensordot(self, other: "COMPSsArray", axes) -> "COMPSsArray":
		...
```

But what if we want to export a `tensordot` function that dispatches to the correct implementation depending on the type of the arguments? One solution would be to expand the `BlockArray.tensordot` method to account for the `COMPSsArray` case, but we want `BlockArray` to be generic, in order to allow multiple array implementations to work with it.

In the standard library, `@functoools.singledispatch` would almost do the job. It dispatches the decorated function on different types, but only on the first argument.

Imagine there is something like `@functools.multidispatch` that dispatches on multiple argument types. Then we could split the implementation of `BlockArray` and `COMPSsArray`. This is what `multimethod` does.

Furthermore, by exporting the functions marked with `@multimethod`, it favours modularity as users can farther extend specialized multimethods with their own implementations and customize the current ones.

```python
from multimethod import multimethod

@multimethod
def tensordot(*args, **kwargs):
	return NotImplemented
```

At `rosnet.array.block`:
```python
from rosnet import dispatch

@dispatch.tensordot.register
def tensordot(a: BlockArray, b: BlockArray, axes) -> BlockArray:
	...
```

At `rosnet.array.compss`:
```python
from rosnet import dispatch
from rosnet.array.block import BlockArray

@dispatch.tensordot.register
def tensordot(a: COMPSsArray, b: COMPSsArray, axes) -> COMPSsArray:
	...
```

And what if we want to compose `BlockArray` and `COMPSsArray` (i.e. `BlockArray[COMPSsArray]`)? We can even do that using `Generic`s.

```python
@dispatch.tensordot.register
def tensordot_block(a: BlockArray[COMPSsArray], b: BlockArray[COMPSsArray], axes, method="commutative"):
	...
```

## Differences with Julia's multiple-dispatch mechanism

- Julia's multiple-dispatch is invariant, while `multimethod` provides ...
- If a multimethod is [type stable](https://docs.julialang.org/en/v1/manual/faq/#man-type-stability), Julia will avoid the virtual table and point directly to the correct implementation. In Python, this is impossible.
