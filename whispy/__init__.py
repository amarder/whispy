"""
Whispy - Python wrapper for whisper.cpp

A fast and efficient Python wrapper for whisper.cpp providing automatic speech recognition.
"""

import ctypes
import os
import pathlib
import sys
from typing import Optional

__version__ = "0.1.0"

class WhispyError(Exception):
    """Base exception for Whispy errors."""
    pass

class LibraryNotFoundError(WhispyError):
    """Raised when the whisper shared library cannot be found."""
    pass

def _find_library() -> pathlib.Path:
    """Find the whisper shared library."""
    # The path to the currently executing package
    package_path = pathlib.Path(__file__).parent.resolve()
    
    # Library name patterns for different platforms
    if sys.platform == "darwin":
        lib_names = ["libwhisper.dylib"]
    elif sys.platform == "win32":
        lib_names = ["whisper.dll", "libwhisper.dll"]
    else:  # Linux and other Unix-like systems
        lib_names = ["libwhisper.so"]
    
    # Search for the library
    for lib_name in lib_names:
        lib_path = package_path / lib_name
        if lib_path.exists():
            return lib_path
    
    # If not found, raise an error with helpful information
    raise LibraryNotFoundError(
        f"Could not find whisper shared library in {package_path}. "
        f"Looked for: {', '.join(lib_names)}. "
        f"Please ensure the library was built correctly."
    )

def _load_library() -> ctypes.CDLL:
    """Load the whisper shared library."""
    lib_path = _find_library()
    
    try:
        return ctypes.CDLL(str(lib_path))
    except OSError as e:
        raise WhispyError(f"Failed to load whisper library from {lib_path}: {e}")

# Global library instance
_libwhisper: Optional[ctypes.CDLL] = None

def get_library() -> ctypes.CDLL:
    """Get the loaded whisper library instance."""
    global _libwhisper
    if _libwhisper is None:
        _libwhisper = _load_library()
    return _libwhisper

def is_library_loaded() -> bool:
    """Check if the whisper library is loaded."""
    return _libwhisper is not None

# Initialize the library on import
try:
    get_library()
    print(f"Successfully loaded whisper library (whispy v{__version__})")
except WhispyError as e:
    print(f"Warning: Failed to load whisper library: {e}")
    print("The library may not be built correctly. Try rebuilding with 'pip install -e .'")

# Export public API
__all__ = [
    "__version__",
    "WhispyError", 
    "LibraryNotFoundError",
    "get_library",
    "is_library_loaded",
]

# Example of how to define functions using the loaded library:
# 
# libwhisper = get_library()
# libwhisper.whisper_init_from_file.argtypes = [ctypes.c_char_p]
# libwhisper.whisper_init_from_file.restype = ctypes.c_void_p
#
# def init_whisper(model_path: str):
#     return libwhisper.whisper_init_from_file(model_path.encode("utf-8"))