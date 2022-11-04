import types
import numpy as np
from numbers import Number
from ..raggedshape import RaggedView

import sys
import numpy as _np


class IndexableArray:

    def __build_data_from_indices_generator(self, indices_generator, size):
        out_data = np.empty(int(size), dtype=self.dtype)
        offset = 0
        for indices in indices_generator:
            n_indices = indices.size
            out_data[offset:offset+n_indices] = self._data[indices]
            offset+=n_indices
        return out_data

    def __getitem__(self, index):
        ret = self._get_row_subset(index, do_split=False)
        if ret == NotImplemented:
            raise NotImplementedError()
        index, shape = ret
        if shape is None:
            return self._data[index]
        if isinstance(index, types.GeneratorType):
            out_data = self.__build_data_from_indices_generator(index, shape.size)
            return self.__class__(out_data, shape)
        return self.__class__(self._data[index], shape)

    def _get_row_col_subset(self, rows, cols):
        if rows is Ellipsis:
            rows = slice(None)
        if cols is Ellipsis:
            cols = slice(None)
        #if np.issubdtype(np.asanyarray(rows).dtype, np.integer) and np.issubdtype(np.asanyarray(cols).dtype, np.integer):
        if np.issubdtype(_np.asanyarray(rows).dtype, np.integer) and np.issubdtype(_np.asanyarray(cols).dtype, np.integer):
            return self._get_element(rows, cols)
        view = self.shape.view_rows(rows)
        view = view.col_slice(cols)
        ret, shape = self._get_view(view)
        if isinstance(rows, Number) or isinstance(cols, Number):
            shape = None
        return ret, shape

    def _get_row_subset(self, index, do_split=False):
        if isinstance(index, tuple):
            if len(index) == 0:
                return slice(None), self.shape
            if len(index) == 1:
                index = index[0]
            elif len(index) > 2:
                index = tuple(i for i in index if i is not Ellipsis)
                return self._get_row_col_subset(index[0], index[1])
            else:
                return self._get_row_col_subset(index[0], index[1])
        if index is Ellipsis:
            return slice(None), self.shape
        elif isinstance(index, Number):
            return self._get_row(index)
        elif isinstance(index, slice):
            return self._get_multiple_rows(index, do_split)
        elif isinstance(index, RaggedView):
            return self._get_view(index)
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            if len(index) == 0:
                index = np.asanyarray(index, dtype=int)
            return self._get_multiple_rows(np.asanyarray(index), do_split)
        elif isinstance(index, IndexableArray):
            if np.issubdtype(index, bool):
                return np.flatnonzero(index.ravel()), None
        else:
            return NotImplemented

    def __setitem__(self, _index, value):
        ret = self._get_row_subset(_index)
        if ret == NotImplemented:
            raise TypeError(f"Invalid index for ragged array {type(_index)}: {_index}")
        index, shape = ret
        if isinstance(index, types.GeneratorType):
            index = np.concatenate(list(index))
        if shape is None:
            self._data[index] = value
        else:
            if isinstance(value, Number):
                self._data[index] = value
            elif isinstance(value, IndexableArray):
                assert value.shape == shape, (value.shape, shape)
                self._data[index] = value._data
            else:
                if isinstance(value, list):
                    value = np.asanyarray(value, dtype=self.dtype)
                if len(value.shape) == 2 and value.shape[-1] == 1:
                    self._data[index] = shape.broadcast_values(value, dtype=self.dtype)
                else:
                    self._data[index] = value.ravel()

    def _get_row(self, index):
        view = self.shape.view(index)
        return slice(view.starts, view.ends), None

    def _get_element(self, row, col):
        row, col = (np.asanyarray(v) for v in (row, col))
        if self._safe_mode and (
            np.any(row >= self.shape.n_rows) or np.any(col >= self.shape.lengths[row])
        ):
            raise IndexError(
                f"Index ({row}, {col}) out of bounds for array with shape {self.shape}"
            )
        col = np.asanyarray(col)
        col = np.where(col < 0, self.shape.lengths[row]+col, col)
        flat_idx = self.shape.starts[row] + col
        return flat_idx, None

    def _get_view(self, view, do_split=False):
        return view.get_flat_indices(do_split)

    def _get_multiple_rows(self, rows, do_split=False):
        return self._get_view(self.shape.view(rows), do_split)

    def subset(self, indexes):
        if indexes.dtype != bool:
            raise NotImplementedError("Can only subset with a boolean RaggedArray")

        data = self.ravel()[indexes.ravel()]
        lengths = np.sum(indexes, axis=-1)
        return self.__class__(data, lengths)
