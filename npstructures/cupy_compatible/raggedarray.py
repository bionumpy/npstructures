import cupy as cp

from ..raggedarray import RaggedArray

class CPRaggedArray(RaggedArray):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _accumulate(*args, **kwargs):
        return NotImplemented

    def _reduce(*args, **kwargs):
        return NotImplemented


    def max(*args, **kwargs):
        return NotImplemented

    def _row_accumulate(self, operator, dtype=None):
        return NotImplemented

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
