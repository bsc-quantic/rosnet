import runpy

submodules = ["autoray"]
for submod in submodules:
    try:
        runpy.run_module(f"{__name__}.{submod}")
    except ImportError:
        pass

__all__ = []
