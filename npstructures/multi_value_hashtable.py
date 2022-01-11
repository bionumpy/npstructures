from .hashtable import HashTable
import numpy as np


class MultiValueHashTable:
    def __init__(self, hash_table, values):
        self._hash_table = hash_table
        self._values = values

    @classmethod
    def from_keys_and_values(cls, keys, values: dict, mod=None):
        hash_table = HashTable(keys, np.arange(len(keys), dtype=np.int64), mod=mod)
        return cls(hash_table, values)

    def __getitem__(self, keys):
        indexes = self._hash_table[keys]
        return {name: value[indexes] for name, value in self._values.items()}
