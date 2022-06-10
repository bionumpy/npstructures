import numpy as np
from numbers import Number
from ..raggedshape import RaggedShape, RaggedView


class IndexableArray:
    def __getitem__(self, index):
        ret = self._get_row_subset(index)
        print(ret)
        if ret == NotImplemented:
            return NotImplemented
        index, shape = ret
        if shape is None:
            return self._data[index]
        return self.__class__(self._data[index], shape)

    def _get_row_col_subset(self, rows, cols):
        if rows is Ellipsis:
            rows = slice(None)
        if cols is Ellipsis:
            cols = slice(None)
        if isinstance(rows, (list, np.ndarray, Number)) and isinstance(
            cols, (list, np.ndarray, Number)
        ):
            return self._get_element(rows, cols)
        view = self.shape.view_rows(rows)
        view = view.col_slice(cols)
        ret, shape = self._get_view(view)
        if isinstance(rows, Number) or isinstance(cols, Number):
            shape = None
        return ret, shape

    def _get_row_subset(self, index):
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
            if True or not ((index.step is None) or index.step == 1):
                return self._get_multiple_rows(index)
            start = index.start
            if start is None:
                start = 0
            return self._get_rows(start, index.stop)
        elif isinstance(index, RaggedView):
            return self._get_view(index)
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            if isinstance(index, list):
                index = np.array(index, dtype=int)
            if index.dtype == bool:
                return self._get_rows_from_boolean(index)
            return self._get_multiple_rows(index)
        else:
            return NotImplemented

    def __setitem__(self, index, value):
        ret = self._get_row_subset(index)
        if ret == NotImplemented:
            return NotImplemented
        index, shape = ret
        if shape is None:
            self._data[index] = value
        else:
            if isinstance(value, Number):
                self._data[index] = value
            elif isinstance(value, IndexableArray):
                assert value.shape == shape
                self._data[index] = value._data
            else:
                self._data[index] = shape.broadcast_values(value, dtype=self.dtype)

    def _get_row(self, index):
        view = self.shape.view(index)
        return slice(view.starts, view.ends), None

    def _get_element(self, row, col):
        if self._safe_mode and (
            np.any(row >= self.shape.n_rows) or np.any(col >= self.shape.lengths[row])
        ):
            raise IndexError(
                f"Index ({row}, {col}) out of bounds for array with shape {self.shape}"
            )
        col = np.where(col < 0, self.shape.lengths[row]+col, col)
        flat_idx = self.shape.starts[row] + col
        print(flat_idx)
        return flat_idx, None

    def _get_rows(self, from_row, to_row):
        if from_row is None:
            from_row = 0
        if to_row is None:
            to_row = len(self)
        if to_row <= from_row:
            return slice(None, None, None), RaggedShape.from_tuple_shape((0, 0))
        data_start = self.shape.view(from_row).starts
        new_shape = self.shape[from_row:to_row]
        data_end = data_start + new_shape.size
        return slice(data_start, data_end), new_shape

    def _get_col_slice(self, col_slice):
        view = self.shape.view_cols(col_slice)
        return view.get_flat_indices()

    def _get_rows_from_boolean(self, boolean_array):
        if boolean_array.size != len(self):
            raise IndexError(
                f"Boolean index {boolean_array} shape does not match number of rows {len(self)}"
            )
        rows = np.flatnonzero(boolean_array)
        return self._get_multiple_rows(rows)

    def _get_view(self, view):
        indices, shape = view.get_flat_indices()
        return indices, shape

    def _get_multiple_rows(self, rows):
        return self._get_view(self.shape.view(rows))
