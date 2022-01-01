from numbers import Number
import numpy as np
import time
from .raggedarray import RaggedArray

class HashTable:
    def __init__(self, keys, values, mod=None, key_dtype=None, value_dtype=None):
        if isinstance(keys, RaggedArray):
            self._keys = keys
            self._mod = len(keys)
            if isinstance(values, Number):
                self._values = values*np.ones_like(self._keys, dtype=value_dtype)
            else:
                self._values = values
                assert isinstance(values, RaggedArray)
        else:
            keys = np.asanyarray(keys, dtype=key_dtype)
            self.dtype = keys.dtype.type
            if mod is None:
                mod = self._get_mod(keys)
            self._mod = mod
            hashes = self._get_hash(keys)
            args = np.argsort(hashes)
            hashes = hashes[args]
            keys = keys[args]
            self._keys = self._build_ragged_array(keys, hashes)
            if isinstance(values, Number):
                self._values = values*np.ones_like(self._keys, dtype=value_dtype)
            else:
                self._values = RaggedArray(values[args], self._keys.shape)

    def _get_mod(self, keys):
        return self.dtype(2*keys.size-1) # TODO: make prime

    def _get_hash(self, keys):
        return keys % self._mod

    def _build_ragged_array(self, keys, hashes):
        unique, counts = np.unique(hashes, return_counts=True)
        lengths = np.zeros(self._mod, dtype=int)
        lengths[unique] = counts
        ra = RaggedArray(keys, lengths)
        return ra

    def __getitem__(self, keys):
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        possible_keys = self._keys[hashes]
        offsets = (possible_keys==keys[:, None]).nonzero()[1]
        assert offsets.size==keys.size, (offsets.size, keys.size)
        return self._values[hashes, offsets]


class Counter(HashTable):
    def __init__(self, keys, values=0, **kwargs):
        super().__init__(keys, values, value_dtype=int, **kwargs)

    def count(self, keys):
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        view = self._keys.shape.view(hashes)
        mask = np.flatnonzero(view.lengths)
        keys = keys[mask]
        hashes = hashes[mask]
        view = view[mask]
        view.empty_removed=True
        rows, offsets = (self._keys[view]==keys[:, None]).nonzero()
        flat_indices = view.ravel_multi_index((rows, offsets))
        self._values.ravel()[:] += np.bincount(flat_indices, minlength=self._values.size)
