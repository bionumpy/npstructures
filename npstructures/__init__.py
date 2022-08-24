"""Top-level package for raggednumpy.
Provides the RaggedArray class to provide numpy-like functionality for
arrays with different row sizes.


"""

__author__ = """Knut Rand"""
__email__ = "knutdrand@gmail.com"
__version__ = "0.1.0"

from . import raggedarray, raggedshape, hashtable

from .raggedarray import RaggedArray
from .raggedshape import RaggedShape, RaggedView, ViewBase
from .hashtable import HashTable, Counter, HashSet
from .npdataclasses import npdataclass, SeqArray, VarLenArray

__all__ = [
    "HashTable",
    "Counter",
    "RaggedShape",
    "RaggedArray",
    "RaggedView",
    "npdataclass",
    "SeqArray",
    "VarLenArray",
]


def set_backend(lib):
    import sys
    import numpy as _np

    from .cupy_compatible.raggedshape import CPRaggedShape, CPRaggedView
    from .cupy_compatible.raggedarray import CPRaggedArray

    sys.modules[__name__].RaggedShape = CPRaggedShape
    sys.modules[__name__].RaggedView = CPRaggedView
    sys.modules[__name__].RaggedArray = CPRaggedArray
    sys.modules[__name__].raggedshape.RaggedShape = CPRaggedShape

    raggedarray.RaggedShape = CPRaggedShape
    #raggedarray.indexablearray.RaggedView = CPRaggedView
    #raggedarray.RaggedShape = CPRaggedShape

    # Explanation for the following changes:
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
    #lib.add = _np.add 
    # Except this does not work

    raggedarray.indexablearray.np = lib
    raggedarray.np = lib
    raggedshape.np = lib
    hashtable.np = lib
