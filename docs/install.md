# Installation

`rosnet` is very easy to install. Just run
```
pip install rosnet
```

```{warning}
Pypi packaging support is underway. Meanwhile, install `rosnet` directly from the Git repository.
```{code-block}
pip install git+https://github.com/UB-Quantic/rosnet.git#wheel=rosnet
```

Current dependencies are:
- `python >= 3.8` Although Python 3.9 or higher is recommended.
- `numpy >= 1.20`
- `multimethod`


## Extras

- `distributed`
- `docs`
- `test`

## Docker

```{note}
A `rosnet` Docker image has not been released yet. Meanwhile, use one of the `mofeing/pycompss` images that already have all the dependencies installed to run distributed.

e.g.
```{code-block}
docker pull mofeing/pycompss:2.10
```

