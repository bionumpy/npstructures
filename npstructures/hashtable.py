import numpy as np
import time
from .raggedarray import RaggedArray
from .indexed_raggedarray import IRaggedArray, IRaggedArrayWithReverse


class HashBase:

    def __init__(self, keys, mod=None, dtype=None):
        if isinstance(keys, RaggedArray):
            self._data = keys
            self._mod = len(keys)
            return

        keys = np.asanyarray(keys, dtype=dtype)
        self.dtype = keys.dtype
        if mod is None:
            mod = self._get_mod(keys)
        self._mod = mod
        hashes = self._get_hash(keys)
        args = np.argsort(hashes)
        hashes = hashes[args]
        keys = keys[args]
        self._data = self._build_ragged_array(keys, hashes)

    def _get_mod(self, keys):
        return self.dtype(2*keys.size-1) # TODO: make prime

    def _get_hash(self, keys):
        return keys % self._mod

    def _build_ragged_array(self, keys, hashes):
        unique, counts = np.unique(hashes, return_counts=True)
        offsets = np.zeros(self._mod+1, dtype=int)
        offsets[unique+1] = counts
        offsets = np.cumsum(offsets)
        ra = RaggedArray(keys, offsets)
        return ra

    def __getitem__(self, keys):
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        possible_keys = self._data[hashes]
        offsets = (possible_keys==keys[:, None]).nonzero()[1]
        assert offsets.size==keys.size, (offsets.size, keys.size)
        return self._values[hashes, offsets]


class Counter(HashBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counts = np.zeros_like(self._data, dtype=int)
        self._values =self._counts

    def count(self, keys):
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        view = self._data.shape.view(hashes)
        mask = np.flatnonzero(view.lengths)
        keys = keys[mask]
        hashes = hashes[mask]
        view = view[mask]
        view.empty_removed=True
        rows, offsets = (self._data[view]==keys[:, None]).nonzero()
        flat_indices = view.ravel_multi_index((rows, offsets))
        self._counts.ravel()[:] += np.bincount(flat_indices, minlength=self._counts.size)


class HashTable(HashBase):
    def __init__(self, keys, values, mod):
        keys = np.asanyarray(keys)
        self._mod = mod
        hashes = self._get_hash(keys)
        args = np.argsort(hashes)
        hashes = hashes[args]
        keys = keys[args]
        self._data = self._build_ragged_array(keys, hashes)
        self._values = RaggedArray(values[args], self._data.shape) #TODO: Avoid using privates

class IHashTable(HashTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = IRaggedArrayWithReverse(self._data)
