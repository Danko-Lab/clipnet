# __init__.py
# Adam He <adamyhe@gmail.com>

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("clipnet")
except PackageNotFoundError:
    __version__ = "unknown"
