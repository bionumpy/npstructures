import cupy as cp
from numbers import Number

from ..hashtable import HashSet, HashTable, Counter

class CPHashSet(HashSet):
    pass

class CPHashTable(HashTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __eq__(self, other):
        t = cp.all(self._keys.ravel() == other._keys.ravel())
        t &= cp.all(self._values.ravel() == other._values.ravel())
        return t

class CPCounter(Counter):
    def __init__(self, keys, values=0, **kwargs):
        super().__init__(keys=keys, values=values, **kwargs)
