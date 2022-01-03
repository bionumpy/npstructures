from numbers import Number
import numpy as np
import time
from .raggedarray import RaggedArray

class HashTable:
    """Enables `dict`-like lookup of values for a predefined set of integer keys

    Provides fast lookup for a predefined set of keys. The set of keys cannot be modified after
    the creation of the `HashTable`. This is in contrast to `dict`, where the set of keys is mutable.

    Parameters
    ----------
    keys : array_like or `RaggedArray`
           The keys for the lookup
    values : array_like or `RaggedArray`
             The corresponding values
    mod : int, optional
          the modulo-value used to create the hashes
    key_dtype : optional
                the datatype to use for keys. (Must be integer-type)
    value_dtype : optional
                  the datatype to use for the values

    Examples
    --------
    >>> table = HashTable([10, 19, 20, 100], [3.14, 2.87, 1.11, 0])
    >>> table[[19, 100]]
    array([2.87, 0.  ])
    """

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
                values = np.asanyarray(values)
                self._values = RaggedArray(values[args], self._keys.shape)

    def _get_indices(self, keys):
        if isinstance(keys, Number):
            h = self._get_hash(keys)
            possible_keys = self._keys[h]
            offset = np.flatnonzero(possible_keys==keys)
            return h, offset
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        possible_keys = self._keys[hashes]
        offsets = (possible_keys==keys[:, None]).nonzero()[1]
        assert offsets.size==keys.size, (offsets.size, keys.size)
        return hashes, offsets

    def __getitem__(self, keys):
        return self._values[self._get_indices(keys)]

    def __setitem__(self, key, value):
        indices = self._get_indices(key)
        self._values[indices] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self._keys.ravel()}, {self._values.ravel()})"

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

    def __eq__(self, other):
        t = np.all(self._keys == other._keys)
        t &= np.all(self._values == other._values)
        return t

class Counter(HashTable):
    """HashTable-based counter to count occurances of a predefined set of integers

    Parameters
    ----------
    keys : array_like or `RaggedArray`
           The elements that are to be counted
    values : array_like or `RaggedArray`, default=0
             Initial counts for the elements

    Examples
    --------
    >>> counter = Counter([1, 12, 123, 1234, 12345])
    >>> counter.count([1, 0, 123, 123, 123, 2, 12345])
    >>> counter
    Counter([    1  1234    12   123 12345], [1 0 0 3 1])
    """

    def __init__(self, keys, values=0, **kwargs):
        super().__init__(keys, values, value_dtype=int, **kwargs)

    def count(self, keys):
        """ Count the occurances of the predefined set of integers

        Parameters
        ----------
        keys : array_like
               The set of integers to count
        """
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
