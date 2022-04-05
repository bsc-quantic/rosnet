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
There are some issues with the `compss/compss` image when using a different Python version. While the COMPSs team fixes it, use the `Dockerfile` located inside `.devcontainer` folder.

```{code-block}
docker build -f .devcontainer/Dockerfile -t compss/compss:rosnet-3.9 .
```

