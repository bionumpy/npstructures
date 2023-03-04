from numbers import Number
import numpy as np
import time
from .raggedarray import RaggedArray

HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for RaggedArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.zeros_like)
def zeros_like(hash_table):
    return hash_table.__class__(hash_table._keys, 0)


class HashTable:
    """Enables `dict`-like lookup of values for a predefined set of integer keys

    Provides fast lookup for a predefined set of keys. The set of keys must be unique
    values andcannot be modified after the creation of the `HashTable`.
    This is in contrast to `dict`, where the set of keys is mutable.
    Indexing both with a single index, or an array_like index is supported. See examples

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

    Attributes
    ----------

    Examples
    --------
    >>> table = HashTable([10, 19, 20, 100], [3.14, 2.87, 1.11, 0])
    >>> table[[19, 100]]
    array([2.87, 0.  ])
    """

    def __init__(
        self, keys, values, mod=None, key_dtype=None, value_dtype=None, safe_mode=True
    ):
        if isinstance(keys, RaggedArray):
            self._keys = keys
            self._mod = len(keys)
            self._values = values
            self.dtype = self._keys.dtype.type
            # assert isinstance(values, RaggedArray)
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
                self._values = values
            else:
                values = np.asanyarray(values)
                self._values = RaggedArray(values[args], self._keys._shape)
        self._safe_mode = safe_mode
        self._value_dtype = (
            value_dtype if isinstance(self._values, Number) else self._values.dtype
        )
        self._key_dtype = self._keys.dtype

    def _get_indices(self, keys):
        if isinstance(keys, Number):
            h = self._get_hash(keys)
            possible_keys = self._keys[h]
            offset = np.flatnonzero(possible_keys == keys)
            return h, offset
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        possible_keys = self._keys[hashes]
        rows, offsets = (possible_keys == keys[:, None]).nonzero()
        if offsets.size < keys.size:
            missing_mask = np.ones(len(keys), dtype=bool)
            missing_mask[rows] = False
            raise IndexError(f'Keys {keys[missing_mask]} missing from hash_table, available: {self._keys.ravel()}')
        return hashes, offsets

    def contains(self, keys):
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        possible_keys = self._keys[hashes]
        rows, offsets = (possible_keys == keys[:, None]).nonzero()
        missing_mask = np.ones(len(keys), dtype=bool)
        missing_mask[rows] = False
        return ~missing_mask

    def __getitem__(self, keys):
        if isinstance(self._values, Number):
            return (
                self._values
                if isinstance(keys, Number)
                else np.full(len(keys), self._values, dtype=self._value_dtype)
            )
        return self._values[self._get_indices(keys)]

    def _fill_values(self):
        if isinstance(self._values, Number):
            self._values = self._values * np.ones_like(
                self._keys, dtype=self._value_dtype
            )
            self._values._safe_mode = False

    def __setitem__(self, key, value):
        self._fill_values()
        indices = self._get_indices(key)
        self._values[indices] = value

    def __repr__(self):
        v = self._values
        if isinstance(v, RaggedArray):
            v = self._values.ravel().tolist()
        return f"{self.__class__.__name__}({self._keys.ravel().tolist()}, {v})"

    def _get_mod(self, keys):
        return self.dtype(2 * keys.size - 1)  # TODO: make prime

    def _get_hash(self, keys):
        return keys % self._mod

    def _build_ragged_array(self, keys, hashes):
        unique, counts = np.unique(hashes, return_counts=True)
        lengths = np.zeros(int(self._mod), dtype=int)
        lengths[unique] = counts
        ra = RaggedArray(keys, lengths)
        return ra

    def __eq__(self, other):
        t = np.all(self._keys == other._keys)
        t &= np.all(self._values == other._values)
        return t

    def __add__(self, other):
        if self._safe_mode and not self._keys.equals(other._keys):
            raise ValueError(
                f"Could not add hash tables with differing keys ({self._keys, other._keys})"
            )
        return HashTable(self._keys, self._values + other._values)

    def __iadd__(self, other):
        if isinstance(other, Number):
            self._values += other
            return self
        if self._safe_mode and not self._keys.equals(other._keys):
            raise ValueError(
                f"Could not add hash tables with differing keys ({self._keys, other._keys})"
            )
        if isinstance(self._values, Number) and not isinstance(other._values, Number):
            self._fill_values()
        self._values += other._values
        return self

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def fill(self, value):
        if isinstance(self._values, Number):
            self._values = value
        else:
            self._values.fill(value)

    def items(self):
        return zip(self._keys.ravel(), self._values.ravel())

    def to_dict(self):
        return dict(zip(self._keys.ravel(), self._values.ravel()))


@implements(np.zeros_like)
def zeros_like(hash_table, dtype=None):
    dtype = hash_table._value_dtype if dtype is None else dtype
    return hash_table.__class__(hash_table._keys, 0, value_dtype=dtype)


@implements(np.ones_like)
def ones_like(hash_table, dtype=None, shape=None):
    dtype = hash_table._value_dtype if dtype is None else dtype
    return hash_table.__class__(hash_table._keys, 1, value_dtype=dtype)


class Counter(HashTable):
    """HashTable-based counter to count occurances of a predefined set of integers

    Parameters
    ----------
    keys : array_like or `RaggedArray`
           The elements that are to be counted
    values : array_like or `RaggedArray`, default=0
             Initial counts for the elements

    Attributes
    ----------

    Examples
    --------
    >>> counter = Counter([1, 12, 123, 1234, 12345])
    >>> counter.count([1, 0, 123, 123, 123, 2, 12345])
    >>> counter
    Counter([1, 1234, 12, 123, 12345], [1, 0, 0, 3, 1])
    >>> counter.count([12, 12, 12, 12, 12])
    Counter([1, 1234, 12, 123, 12345], [1, 0, 5, 3, 1])
    """

    def __init__(self, keys, values=0, **kwargs):
        # value_dtype=int
        if not ("value_dtype" in kwargs and kwargs["value_dtype"] is not None):
            kwargs["value_dtype"] = int
        super().__init__(keys, values, **kwargs)
        self._keys._safe_mode = False
        if isinstance(self._values, RaggedArray):
            self._values._safe_mode = False

    def count(self, keys):
        """Count the occurances of the predefined set of integers.

        Updates the counts in the Counter with the number of occurances
        of each of its keys in `keys`.

        Parameters
        ----------
        keys : array_like
               The set of integers to count
        """
        t = time.time()
        keys = np.asanyarray(keys, dtype=self._key_dtype)
        hashes = self._get_hash(keys)
        view = self._keys._shape.view(hashes)
        mask = np.flatnonzero(view.lengths)
        keys = keys[mask]
        hashes = hashes[mask]
        view = view[mask]
        view.empty_removed = True
        rows, offsets = (self._keys[view] == keys[:, None]).nonzero()
        if not rows.size:
            return
        flat_indices = view.ravel_multi_index((rows, offsets))
        if isinstance(self._values, Number):
            if self._values == 0:
                self._values = RaggedArray(
                    np.bincount(flat_indices, minlength=self._keys.size),
                    self._keys._shape,
                    dtype=self._value_dtype,
                    safe_mode=False,
                )
            else:
                self._values = RaggedArray(
                    self._values + np.bincount(flat_indices, minlength=self._keys.size),
                    self._keys._shape,
                    dtype=self._value_dtype,
                )
        else:
            self._values.ravel()[:] += np.bincount(
                flat_indices,
                minlength=self._values.size
            ).astype(self._value_dtype)


class HashSet(HashTable):
    def __init__(self, keys, mod=None, key_dtype=None):
        super().__init__(keys, 0, mod, key_dtype)

    def contains(self, keys):
        if isinstance(keys, Number):
            h = self._get_hash(keys)
            possible_keys = self._keys[h]
            return np.any(possible_keys == keys)
        keys = np.asanyarray(keys)
        hashes = self._get_hash(keys)
        possible_keys = self._keys[hashes]
        return np.any(possible_keys == keys[:, None], axis=-1)
