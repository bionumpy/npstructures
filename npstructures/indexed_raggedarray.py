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
            data, row_lens, index_lookup = self.from_array_list(data)
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

    def _broadcast_rows(self, _values):
        assert _values.shape == (self._row_lens.size, 1)
        _values = _values.ravel()
        values = np.empty_like(_values)
        values[self._index_lookup] = values
        value_diffs = np.diff(values)

        
        offsets = np.empty_like(self._row_lens)
        offsets[self._index_lookup] = self._row_lens
        offsets = np.cumsum(offsets)

        broadcasted = np.zeros_like(self._data._data)
        broadcasted[offsets[:-1]] = value_diffs
        broadcasted[0] = values[0]
        func = np.logical_xor if values.dtype==bool else np.add
        broadcasted = RaggedArray(func.accumulate(broadcasted), self._data._offsets)
        return self.__class__(broadcasted, self._row_lens, self._index_lookup)

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
            size = self._data[row_len-1].size
            if size == 0:
                continue
            indexes = np.nonzero(self._row_lens==row_len)
            local_inputs = [getter.get(row_len, indexes) for getter in getters]
            result = ufunc(*local_inputs, **kwargs).ravel()
            assert result.size == size, (size, result.size, self._data[row_len], result, row_len,"\n", local_inputs, ufunc(*local_inputs))
            new_data[cur_offset:cur_offset+size] = result
            cur_offset += size
        new_data = RaggedArray(new_data, self._data._offsets)
        return self.__class__(new_data, self._row_lens, self._index_lookup)
