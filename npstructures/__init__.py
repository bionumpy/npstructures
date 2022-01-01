"""Top-level package for raggednumpy.
Provides the RaggedArray class to provide numpy-like functionality for 
arrays with different row sizes.


"""

__author__ = """Knut Rand"""
__email__ = 'knutdrand@gmail.com'
__version__ = '0.1.0'

from .raggedarray import RaggedArray
from .raggedarray import RaggedShape, RaggedView
# from .indexed_raggedarray import IRaggedArray, IRaggedArrayWithReverse
from .hashtable import HashTable, Counter
