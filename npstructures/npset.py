from .hashtable import HashTable
import numpy as np


class NpSet:
    def __init__(self, values):
        if isinstance(values, HashTable):
            self._values = values
        else:
            unique_values = np.unique(values)
            self._values = HashTable(unique_values, np.ones(len(unique_values)))

    def __len__(self):
        return self

    def __contains__(self, item):
        return len(self._values[self._values.dtype(item)]) > 0

