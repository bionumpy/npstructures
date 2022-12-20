"""Top-level package for raggednumpy.
Provides the RaggedArray class to provide numpy-like functionality for
arrays with different row sizes.


"""

__author__ = """Knut Rand"""
__email__ = "knutdrand@gmail.com"
__version__ = '0.2.9'

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

    from .cupy_compatible.raggedshape import CPRaggedShape, CPRaggedView, CPRaggedView2
    from .cupy_compatible.raggedarray import CPRaggedArray
    from .cupy_compatible.hashtable import CPHashTable, CPCounter, CPHashSet
    from .cupy_compatible.bitarray import CPBitArray
    from .cupy_compatible.util import cp_unsafe_extend_right, cp_unsafe_extend_left

    sys.modules[__name__].RaggedShape = CPRaggedShape
    sys.modules[__name__].RaggedView = CPRaggedView
    sys.modules[__name__].RaggedArray = CPRaggedArray
    #sys.modules[__name__].raggedshape.RaggedShape = CPRaggedShape
    sys.modules[__name__].HashTable = CPHashTable
    sys.modules[__name__].Counter = CPCounter
    sys.modules[__name__].HashSet = CPHashSet
    #sys.modules[__name__].hashtable.RaggedArray = CPRaggedArray
    sys.modules[__name__].bitarray.BitArray = CPBitArray

    raggedarray.RaggedShape = CPRaggedShape
    raggedarray.unsafe_extend_left = cp_unsafe_extend_left
    hashtable.RaggedArray = CPRaggedArray

    raggedarray.indexablearray.np = lib
    raggedarray.np = lib
    raggedshape.np = lib
    hashtable.np = lib

    from .raggedarray import indexablearray
    indexablearray.RaggedView = CPRaggedView

    # Explanation for the following changes:
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
    #import numpy as _np
    #lib.add = _np.add 
    # Except this does not work
