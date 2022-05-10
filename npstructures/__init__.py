"""Top-level package for raggednumpy.
Provides the RaggedArray class to provide numpy-like functionality for 
arrays with different row sizes.


"""

__author__ = """Knut Rand"""
__email__ = 'knutdrand@gmail.com'
__version__ = '0.1.0'


from .raggedarray import RaggedArray
from .raggedshape import RaggedShape, RaggedView
from .hashtable import HashTable, Counter
from .npdataclasses import npdataclass, SeqArray, VarLenArray
__all__ = ["HashTable", "Counter",
           "RaggedShape", "RaggedArray", "RaggedView",
           "npdataclass", "SeqArray", "VarLenArray"]
