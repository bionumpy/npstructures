import numpy as np

from .raggedarray import RaggedArray

class IRaggedArray(RaggedArray):

    def __init__(self, data, row_lens=None, index_lookup=None):
        if row_lens is None:
            data, row_lens, index_lookup = self.from_array_list(data)
        self._data = data
        assert np.all(row_lens.shape == index_lookup.shape), (row_lens.shape, index_lookup.shape)
        self._row_lens = row_lens
        self._index_lookup = index_lookup
    
    def equals(self, other):
        t = np.all(self._row_lens == other._row_lens)
        t &= np.all(self._index_lookup  == other._index_lookup)
        t &= np.all(self._data==other._data)
        return t

    def __repr__(self):
        return f"IRaggedArray({repr(self._data)}, {repr(self._row_lens)}, {repr(self._index_lookup)})"

    def to_array_list(self):
        return [self._get_data_for_rowlen(row_len)[index_lookup]
                for row_len, index_lookup 
                in zip(self._row_lens, self._index_lookup)]

    @classmethod
    def from_array_list(cls, array_list):
        row_lens = np.array([len(a) for a in array_list])
        counts = np.bincount(row_lens)
        index_lookup = np.empty(row_lens.size, dtype=int)
        data = np.empty(np.sum(row_lens))
        offsets = [0]
        cur_offset = 0
        index_lookup[row_lens==0] = 0
        for row_len, count in enumerate(counts[1:], 1):
            d = np.flatnonzero(row_lens==row_len)
            index_lookup[d] = np.arange(count)
            data[cur_offset:(cur_offset+count*row_len)] = np.array([array_list[i] for i in d]).ravel()
            cur_offset += count*row_len
            offsets.append(cur_offset)
            #data.append(np.array([array_list[i] for i in d]))
        assert (row_lens.shape==index_lookup.shape), (row_lens.shape, index_lookup.shape)
        
        return RaggedArray(data, np.array(offsets)), row_lens, index_lookup

    def _get_data_for_rowlen(self, row_len):
        if row_len == 0:
            return np.zeros((1, 0))

        chunk = self._data[row_len-1]
        if chunk.size==0:
            return np.zeros((0, row_len))
        return chunk.reshape(-1, row_len)

    def _get_row(self, index):
        return self._get_data_for_rowlen(self._row_lens[index])[self._index_lookup[index]]

    def _get_rows(self, from_row, to_row, step=None):
        if step is None:
            step = 1
        indices = np.arange(from_row, to_row, step)
        return self._get_multiple_rows(indices)

    def _get_multiple_rows(self, rows):
        row_lens = self._row_lens[rows]
        indexes = self._index_lookup[rows]
        data = np.empty(np.sum(row_lens))
        cur_offset = 0
        offsets = [cur_offset]
        for row_len in range(1, np.max(row_lens)+1):
            d = np.flatnonzero(row_lens == row_len)
            data[cur_offset:cur_offset+row_len*d.size] = self._get_data_for_rowlen(row_len)[indexes[d]].ravel()
            cur_offset += row_len*d.size
            offsets.append(cur_offset)
            indexes[d] = np.arange(d.size)
        indexes[row_lens==0] = 0
        return self.__class__(RaggedArray(data, np.array(offsets)), row_lens, indexes)

                              

