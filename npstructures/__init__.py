"""Top-level package for raggednumpy.
Provides the RaggedArray class to provide numpy-like functionality for
arrays with different row sizes.


"""

__author__ = """Knut Rand"""
__email__ = "knutdrand@gmail.com"
__version__ = "0.1.0"

from . import raggedarray, raggedshape, hashtable

from .raggedarray import RaggedArray
from .raggedarray.raggedslice import ragged_slice
from .raggedshape import RaggedShape, RaggedView
from .hashtable import HashTable, Counter, HashSet
from .npdataclasses import npdataclass, VarLenArray

__all__ = [
    "HashTable",
    "Counter",
    "RaggedShape",
    "RaggedArray",
    "RaggedView",
    "npdataclass",
    "VarLenArray",
    "ragged_slice"
]


def set_backend(lib):
    import sys

    from .cupy_compatible.raggedshape import CPRaggedShape
    from .cupy_compatible.raggedarray import CPRaggedArray

    sys.modules[__name__].RaggedShape = CPRaggedShape
    sys.modules[__name__].RaggedArray = CPRaggedArray

    raggedarray.np = lib
    raggedshape.np = lib
    hashtable.np = lib
