import numpy as np
from numbers import Number

from .raggedarray import RaggedArray

class _ScalarWrapper:
    def __init__(self, scalar):
        self.scalar = scalar
    def get(self, row_len, indices):
        return self.scalar

class _ArrayWrapper:
    def __init__(self, array):
        self._array = array

    def get(self, row_len, indices):
        return self._array[indices]

class _IRaggedWrapper:
    def __init__(self, ragged):
        self._ragged = ragged

    def get(self, row_len, indices):
        return self._ragged._get_data_for_rowlen(row_len)


class IRaggedArray(RaggedArray):
    def __init__(self, data, row_lens=None, index_lookup=None):
        if row_lens is None:
            if isinstance(data, list):
                data, row_lens, index_lookup = self.from_array_list(data)
            elif isinstance(data, RaggedArray):
                data, row_lens, index_lookup = self.from_ragged_array(data)
        self._data = data
        assert np.all(row_lens.shape == index_lookup.shape), (row_lens.shape, index_lookup.shape)
        self._row_lens = row_lens
        self._index_lookup = index_lookup
    
    def equals(self, other):
        t = np.all(self._row_lens == other._row_lens)
        t &= np.all(self._index_lookup  == other._index_lookup)
        t &= self._data.equals(other._data)
        return t

    def __repr__(self):
        return f"IRaggedArray({repr(self._data)}, {repr(self._row_lens)}, {repr(self._index_lookup)})"

    def __iter__(self):
        return (self._get_data_for_rowlen(row_len)[index_lookup]
                for row_len, index_lookup 
                in zip(self._row_lens, self._index_lookup))

    def __len__(self):
        return self._row_lens.size

    def to_array_list(self):
        return list(self)

    @classmethod
    def from_ragged_array(cls, ragged_array):
        row_lens = ragged_array.shape.lengths
        args = np.argsort(row_lens, kind="stable")
        data = ragged_array[args]._data
        max_row_len = np.max(row_lens)
        index_lookup = np.zeros(row_lens.size, dtype=int)
        counts = []
        for row_len in range(1, max_row_len+1):
            idxs = np.flatnonzero(row_lens==row_len)
            index_lookup[idxs] = np.arange(idxs.size)
            counts.append(idxs.size*row_len)

        return RaggedArray(data, counts), row_lens, index_lookup

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
        assert (row_lens.shape==index_lookup.shape), (row_lens.shape, index_lookup.shape)
        
        return RaggedArray(data, np.diff(offsets)), row_lens, index_lookup

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
        return self.__class__(RaggedArray(data, np.diff(offsets)), row_lens, indexes)

    def _get_row_lengths(self):
        return range(1, len(self._data)+1)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not method == "__call__":
            return NotImplemented

        getters = []
        for input in inputs:
            if isinstance(input, Number):
                getters.append(_ScalarWrapper(input))
            elif isinstance(input, np.ndarray):
                getters.append(_ArrayWrapper(input))
            elif isinstance(input, IRaggedArray):
                getters.append(_IRaggedWrapper(input))
            else:
                return NotImplemented

        new_data = np.zeros(self._data.size, self._data.dtype)
        cur_offset = 0
        for row_len in self._get_row_lengths():
            size = self._get_data_for_rowlen(row_len).size
            if size == 0:
                continue
            indexes = np.nonzero(self._row_lens==row_len)
            local_inputs = [getter.get(row_len, indexes) for getter in getters]
            result = ufunc(*local_inputs, **kwargs).ravel()
            assert result.size == size, (size, result.size, self._data[row_len], result, row_len,"\n", local_inputs, ufunc(*local_inputs))
            new_data[cur_offset:cur_offset+size] = result
            cur_offset += size
        new_data = RaggedArray(new_data, self._data.shape)
        return self.__class__(new_data, self._row_lens, self._index_lookup)


class IRaggedArrayWithReverse(IRaggedArray):
    def __init__(self, data, row_lens=None, index_lookup=None, reverse=None):
        super().__init__(data, row_lens, index_lookup)
        if reverse is None:
            reverse = self._get_reverse()
        self._reverse = reverse

    def _get_reverse(self):
        reverse = []
        for row_len in self._get_row_lengths():
            reverse.append(np.flatnonzero(self._row_lens==row_len))
        return reverse
            
    def nonzero(self):
        rows = []
        offsets = []
        for row_len, reverse in zip(self._get_row_lengths(), self._reverse):
            row, offset = self._get_data_for_rowlen(row_len).nonzero()
            rows.append(reverse[row])
            offsets.append(offset)
        rows = np.concatenate(rows)
        args = np.argsort(rows, kind="mergesort")
        return rows[args], np.concatenate(offsets)[args]

