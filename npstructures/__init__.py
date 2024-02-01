"""Top-level package for raggednumpy.
Provides the RaggedArray class to provide numpy-like functionality for
arrays with different row sizes.


"""

__author__ = """Knut Rand"""
__email__ = "knutdrand@gmail.com"
__version__ = '0.2.16'

from . import raggedarray, raggedshape, hashtable

from .raggedarray import RaggedArray
from .raggedarray.raggedslice import ragged_slice
from .raggedshape import RaggedShape, RaggedView
from .hashtable import HashTable, Counter, HashSet
from .bitarray import BitArray
from .npdataclasses import npdataclass, VarLenArray
from .runlengtharray import RunLength2dArray, RunLengthArray, RunLengthRaggedArray
__all__ = [
    "HashTable",
    "Counter",
    "RaggedShape",
    "RaggedArray",
    "RaggedView",
    "npdataclass",
    "VarLenArray",
    "ragged_slice",
    "RunLenghtArray",
    "RunLength2dArray",
    "RunLengthRaggedArray"
]


def set_backend(lib):
    import sys
    from . import util
    util.np.set_backend(lib)
