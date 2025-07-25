import pathlib

class Path(pathlib.PosixPath):
    '''Wrapper for pathlib.Path to ensure string representation is used in Spectrum class.'''
    def endswith(self, suffix: str) -> bool:
        return str(self).endswith(suffix)
    
    def split(self, sep: str | None = None) -> list[str]:
        return str(self).split(sep)
    
    def __add__(self, other: str) -> 'Path':
        return Path(str(self) + other)
    
    def __radd__(self, other: str) -> 'Path':
        return Path(other + str(self))