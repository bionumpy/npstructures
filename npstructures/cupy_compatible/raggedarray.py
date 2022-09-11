import cupy as cp
import numpy as np
from numbers import Number

from ..raggedarray import RaggedArray
from .raggedshape import CPRaggedShape

class CPRaggedArray(RaggedArray):
    def __init__(self, data, *args, **kwargs):
        if isinstance(data, (list, np.ndarray, cp.ndarray)):
            super().__init__(data, *args, **kwargs)
        else:
            data = data._ndarray
            super().__init__(data, *args, **kwargs)

    def astype(self, dtype):
        return CPRaggedArray(self._data.astype(dtype), self.shape)

    def _accumulate(*args, **kwargs):
        return NotImplemented

    def _reduce(*args, **kwargs):
        return NotImplemented

    def max(*args, **kwargs):
        return NotImplemented

    def _row_accumulate(self, operator, dtype=None):
        return NotImplemented

    def _get_row_col_subset(self, rows, cols):
        if rows is Ellipsis:
            rows = slice(None)
        if cols is Ellipsis:
            cols = slice(None)

        if isinstance(rows, cp.ndarray):
            rows = cp.asnumpy(rows)
        if isinstance(cols, cp.ndarray):
            cols = cp.asnumpy(cols)

        if np.issubdtype(np.asanyarray(rows).dtype, np.integer) and np.issubdtype(np.asanyarray(cols).dtype, np.integer):
            return self._get_element(rows, cols)
        view = self.shape.view_rows(rows)
        view = view.col_slice(cols)
        ret, shape = self._get_view(view)
        if isinstance(rows, Number) or isinstance(cols, Number):
            shape = None
        return ret, shape

    def _get_element(self, row, col):
        # cp.any and np.any seem to work differently here.
        # cp.any does not accept scalars for a, but np.any does.
        # row and col are (usually?) scalars here
        """
        if self._safe_mode and (
            cp.any(row >= self.shape.n_rows) or cp.any(col >= self.shape.lengths[row])
        ):
            raise IndexError(
                f"Index ({row}, {col}) out of bounds for array with shape {self.shape}"
            )
        """

        col = cp.asanyarray(col)
        col = cp.where(col < 0, self.shape.lengths[row]+col, col)
        flat_idx = self.shape.starts[row] + col
        return flat_idx, None
