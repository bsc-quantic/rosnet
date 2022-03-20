import importlib.util
import modulefinder
import sys
from typing import Sequence


def get_modules(name: str) -> Sequence[str]:
    finder = modulefinder.ModuleFinder()
    finder.load_file(name)
    return list(finder.modules.keys())


if __name__ == "__main__":
    filename = sys.argv[0]
    del sys.argv[0]

    with open(filename, "r") as handler:
        code = handler.read()
        exec(code)
