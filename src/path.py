import os
import pathlib

if os.name == "nt":
    _PathBase = pathlib.WindowsPath
else:
    _PathBase = pathlib.PosixPath


class Path(_PathBase):
    """
    Wrapper for pathlib.Path to ensure string representation is used
    in certain operations, such as endswith and split.
    """
    def endswith(self, suffix: str) -> bool:
        return str(self).endswith(suffix)

    def split(self, sep: str | None = None) -> list[str]:
        return str(self).split(sep)

    def __add__(self, other: str) -> "Path":
        return Path(str(self) + other)

    def __radd__(self, other: str) -> "Path":
        return Path(other + str(self))