import numpy as np

from .raggedarray import RaggedArray
from .indexed_raggedarray import IRaggedArray, IRaggedArrayWithReverse

class HashTable:

    def __init__(self, keys, values, mod):
        keys = np.asanyarray(keys)
        values = np.asanyarray(values)
        self._mod = mod
        hashes = self._get_hash(keys)
        args = np.argsort(hashes)
        hashes = hashes[args]
        keys = keys[args]
        values = values[args]
        self._data, self._values = self._build_ragged_arrays(keys, hashes, values)

    def _build_ragged_arrays(self, keys, hashes, values):
        unique, counts = np.unique(hashes, return_counts=True)
        offsets = np.zeros(self._mod+1, dtype=int)
        offsets[unique+1] = counts
        offsets = np.cumsum(offsets)
        ra = RaggedArray(keys, offsets)
        va = RaggedArray(values, offsets)
        return IRaggedArrayWithReverse(ra), va

    def __getitem__(self, keys):
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        possible_keys = self._data[hashes]
        offsets = (possible_keys==keys[:, None]).nonzero()[1]
        return self._values[hashes, offsets]

    def _get_hash(self, keys):
        return keys % self._mod
