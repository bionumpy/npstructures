import numpy as np

from .raggedarray import RaggedArray

class IRaggedArray(RaggedArray):

    def __init__(self, data, row_lens=None, index_lookup=None):
        if row_lens is None:
            data, row_lens, index_lookup = self.from_array_list(data)
        print("###", data)
        self._data = data
        assert np.all(row_lens.shape == index_lookup.shape), (row_lens.shape, index_lookup.shape)
        self._row_lens = row_lens
        self._index_lookup = index_lookup
    
    def equals(self, other):
        t = np.all(self._row_lens == other._row_lens)
        t &= np.all(self._index_lookup  == other._index_lookup)
        t &= all(np.all(sd.ravel()==od.ravel()) for sd, od in zip(self._data, other._data))
        return t

    def __repr__(self):
        return f"IRaggedArray({repr(self._data)}, {repr(self._row_lens)}, {repr(self._index_lookup)})"

    def to_array_list(self):
        return [self._data[row_len][index_lookup]
                for row_len, index_lookup 
                in zip(self._row_lens, self._index_lookup)]

    @classmethod
    def from_array_list(cls, array_list):
        row_lens = np.array([len(a) for a in array_list])
        counts = np.bincount(row_lens)
        index_lookup = np.empty(row_lens.size, dtype=int)
        data = []
        for row_len, count in enumerate(counts):
            d = np.flatnonzero(row_lens==row_len)
            index_lookup[d] = np.arange(count)
            data.append(np.array([array_list[i] for i in d]))
        assert (row_lens.shape==index_lookup.shape), (row_lens.shape, index_lookup.shape)
        return data, row_lens, index_lookup

    def _get_row(self, index):
        return self._data[self._row_lens[index]][self._index_lookup[index]]

    def _get_rows(self, from_row, to_row, step=None):
        if step is None:
            step = 1
        indices = np.arange(from_row, to_row, step)
        return self._get_multiple_rows(indices)

    def _get_multiple_rows(self, rows):
        row_lens = self._row_lens[rows]
        indexes = self._index_lookup[rows]
        print(row_lens.shape, indexes.shape)
        # new_indexes = np.empty_like(indexes)
        data = []
        for row_len in range(np.max(row_lens)+1):
            d = np.flatnonzero(row_lens == row_len)
            data.append(self._data[row_len][indexes[d]])
            indexes[d] = np.arange(d.size)
        return self.__class__(data, row_lens, indexes)
