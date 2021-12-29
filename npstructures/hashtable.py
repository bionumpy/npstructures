import numpy as np

from .raggedarray import RaggedArray

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
        self._data = self._build_ragged_array(keys, hashes)
        self._values = values

    def _build_ragged_array(self, keys, hashes):
        unique, counts = np.unique(hashes, return_counts=True)
        offsets = np.zeros(self._mod+1, dtype=int)
        offsets[unique+1] = counts
        offsets = np.cumsum(offsets)
        return RaggedArray(keys, offsets)

    def __getitem__(self, index):
        index = np.asanyarray(index)
        hashes = self._get_hash(index)
        possible_keys = self._data[hashes]
        offsets = (possible_keys==index[:, None]).nonzero()[1]
        idxs = self._data._offsets[hashes]+offsets
        return self._values[idxs]

    def _get_hash(self, keys):
        return keys % self._mod
