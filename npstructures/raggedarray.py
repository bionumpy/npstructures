import numpy as np

class RaggedArray:
    def __init__(self, data, offsets):
        self._data = data
        self._offsets = offsets

        self._row_starts = self._offsets[:-1]
        self._row_ends = self._offsets[1:]

    def __repr__(self):
        return f"RaggedArray({repr(self._data)}, {repr(self._offsets)})"

    def __str__(self):
        return str(self.to_array_list())

    def equals(self, other):
        return np.all(self._data==other._data) and np.all(self._offsets == other._offsets)

    def to_array_list(self):
        return [self._data[start, end] for start, end in zip(self._row_starts, self._row_ends)]

    @classmethod
    def from_array_list(cls, array_list):
        offsets = np.cumsum([0] + [len(array) for array in array_list])
        data_size = offsets[-1]
        data = np.array([element for array in array_list for element in array]) # This can be done faster
        return cls(data, offsets)

    def _get_row(self, index):
        assert 0 <= index < self._row_starts.size
        return self._data[self._row_starts[index]:self._row_ends[index]]

    def _get_rows(self, from_row, to_row):
        data_start = self._offsets[from_row]
        data_end = self._offsets[to_row]
        new_data = self._data[data_start:data_end]
        new_offsets = self._offsets[from_row:to_row+1]
        return self.__class__(new_data, new_offsets-data_start)

    def _get_rows_from_boolean(self, boolean_array):
        rows = np.flatnonzero(boolean_array)
        return self._get_multiple_rows(rows)

    def _get_multiple_rows(self, rows):
        """ This is quite slow. Requires the building of a full boolean mask """
        starts = self._row_starts[rows]
        ends = self._row_ends[rows]
        mask = self._build_mask(starts, ends)
        new_data = self._data[mask]
        new_offsets = np.insert(np.cumsum(ends-starts), 0, 0)
        return self.__class__(new_data, new_offsets)

    def _get_element(self, row, col):
        flat_idx = self._row_starts[row] + col
        assert flat_idx < self._row_ends[row]
        return self._data[flat_idx]

    def _build_mask(self, starts, ends):
        full_boolean_array = np.zeros(self._data.size+1, dtype=bool)
        full_boolean_array[starts] = True
        full_boolean_array[ends] ^= True
        return np.logical_xor.accumulate(full_boolean_array)[:-1]

    def __getitem__(self, index):
        if isinstance(index, tuple):
            assert len(index)==2
            return self._get_element(index[0], index[1])
        elif isinstance(index, int):
            return self._get_row(index)
        elif isinstance(index, slice):
            assert (index.step is None) or index.step==1
            return self._get_rows(index.start, index.stop)
        elif isinstance(index, list):
            return self._get_multiple_rows(index)
        elif index.dtype==bool:
            return self._get_rows_from_boolean(index)
