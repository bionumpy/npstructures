import cupy as cp
import numpy as np
from numbers import Number
from dataclasses import dataclass

from ..raggedshape import RaggedShape, RaggedRow, RaggedView, RaggedView2

class CPRaggedShape(RaggedShape):
    def __init__(self, codes, is_coded=False):
        #print(f"{codes.dtype}=?{self._dtype}")
        assert codes.dtype == self._dtype
        codes = cp.asanyarray(codes, dtype=self._dtype)
        super().__init__(codes, is_coded=is_coded)

    # From ViewBase
    def index_array(self):
        """Return an array of broadcasted row indices"""
        diffs = cp.zeros(int(self.size) + 1, dtype=self._dtype)
        diffs = cp.bincount(self.starts[1:], minlength=int(self.size) + 1)
        # diffs[self.starts[1:]] = 1
        return cp.cumsum(diffs)[:-1]

    def view(self, indices, squeeze=True):
        """Return a view of a subset of rows

        Return a view with row information for the row given by `indices`

        Parameters
        ----------
        indices : index_like
            Used to index the rows

        Returns
        -------
        RaggedView
            RaggedView containing information to find the rows specified by `indices`
        """
        if squeeze and isinstance(indices, Number):
            return CPRaggedRow(self._index_rows(indices))
        # self._codes.view(np.uint64)[indices])
        # return RaggedView(self._codes.view(np.uint64)[indices])
        return CPRaggedView(self._index_rows(indices))

    def view_rows(self, indices):
        idx = self._index_rows(indices).reshape(-1, 2)
        return CPRaggedView2(idx[..., 0], idx[..., 1])

    def _get_accumulation_func(self, dtype):
        return np.logical_xor.accumulate if dtype == bool else np.add.accumulate

    def _raw_broadcast(self, values, dtype=None):
        """Currently moves broadcast_builder from device to host to perform 
        np.bitwise_xor.accumulate then moves it back to device before return"""
        dtypes = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64, 16: "uint128"}
        orig_dtype = values.dtype
        values = values.view(dtypes[orig_dtype.itemsize])
        broadcast_builder = cp.zeros(int(self.size) + 1, dtype=values.dtype)
        broadcast_builder[self.ends[::-1]] ^= values[::-1]
        broadcast_builder[0] = 0
        broadcast_builder[self.starts] ^= values
        accumulation_func = np.bitwise_xor.accumulate
        #return cp.array(accumulation_func(cp.asnumpy(broadcast_builder[:-1])).view(orig_dtype))
        return cp.array(accumulation_func(cp.asnumpy(broadcast_builder)[:-1]).view(orig_dtype))

    def _broadcast_values_fast(self, values, dtype=None):
        values = values.ravel()
        broadcast_builder = cp.zeros(int(self.size), dtype=dtype)
        broadcast_builder[self.starts[1:]] = cp.diff(values)
        broadcast_builder[0] = values[0]
        accumulation_func = self._get_accumulation_func(values.dtype)

        broadcast_builder = cp.asnumpy(broadcast_builder)
        accumulation_func(broadcast_builder, out=broadcast_builder)
        broadcast_builder = cp.asanyarray(broadcast_builder)

        return broadcast_builder

class CPRaggedRow(RaggedRow):
    def view_cols(self, idx):
        if isinstance(idx, Number):
            if idx >= 0:
                return CPRaggedView(self.starts + idx, cp.ones_like(self.lengths))
            return CPRaggedView(self.ends + idx, cp.ones_like(self.lengths))
        col_slice = idx
        starts = self.starts
        lengths = self.lengths
        ends = self.ends
        if col_slice.step is not None and col_slice.step < 0:
            col_slice = slice(None if col_slice.stop is None else col_slice.stop+1,
                              None if col_slice.start is None else col_slice.start+1,
                              col_slice.step)

        if col_slice.start is not None:
            if col_slice.start >= 0:
                starts = starts + cp.minimum(lengths, col_slice.start)
            else:
                starts = starts + cp.maximum(lengths + col_slice.start, 0)
        if col_slice.stop is not None:
            if col_slice.stop >= 0:
                ends = cp.minimum(self.starts + col_slice.stop, ends)
            else:
                ends = cp.maximum(self.ends + col_slice.stop, starts)
        
        return CPRaggedView(cp.array([starts]), cp.array([np.maximum(0, ends - starts)]), col_slice.step)


class CPRaggedView(RaggedView):
    def __getitem__(self, index):
        if isinstance(index, Number):
            return CPRaggedRow(
                    self._index_rows(index)
            )

        return self.__class__(
                self._index_rows(index)
        )

    def get_shape(self):
        """Return the shape of a ragged array containing the view's rows
        Returns
        -------
        RaggedShape
            The shape of a ragged array consisting of the rows in this view
        """
        if not self.n_rows:
            return CPRaggedShape(self._codes, is_coded=True)

        codes = self._codes.copy()
        if self._step is not None:
            codes[1::2] //= cp.abs(self._step)
        cp.cumsum(codes[1:-1:2], out=codes[2::2])
        codes[0] = 0
        return CPRaggedShape(codes, is_coded=True)

    def _build_indices(self, shape):
        step = 1 if self._step is None else self._step
        index_builder = cp.full(int(shape.size) + 1, step, dtype=self._dtype)
        if (step >= 0):
            index_builder[shape.ends[::-1]] = 1 - self.ends[::-1]
            index_builder[0] = 0
            index_builder[shape.starts] += self.starts
        else:
            index_builder[shape.ends[::-1]] = - (self.starts[::-1]+1)
            index_builder[0] = 0
            index_builder[shape.starts] += (self.ends-1)
        np.cumsum(index_builder, out=index_builder)
        return index_builder[:-1], shape

    def get_flat_indices(self, do_split=False):
        """Return the indices into a flattened array

        Return the indices of all the elements in all the
        rows in this view

        Returns
        -------
        array
        """
        if not self.n_rows:
            return cp.ones(0, dtype=self._dtype), self.get_shape()

        if self.empty_rows_removed():
            return self._get_flat_indices_fast()
        shape = self.get_shape()
        chunk_size = 100000
        # chunk_size = 50000000000
        if do_split and self.starts.size > chunk_size:
            slices = (slice(i*chunk_size, (i+1)*chunk_size) for i in range((len(self.starts)-1)//chunk_size+1))
            return (self[s]._build_indices(shape[s])[0] for s in slices), shape

        return self._build_indices(shape)
        step = 1 if self._step is None else self._step
        index_builder = cp.full(int(shape.size + 1), step, dtype=self._dtype)
        if (step >= 0):
            index_builder[shape.ends[::-1]] = 1 - self.ends[::-1]
            index_builder[0] = 0
            index_builder[shape.starts] += self.starts
        else:
            index_builder[shape.ends[::-1]] = - (self.starts[::-1]+1)
            index_builder[0] = 0
            index_builder[shape.starts] += (self.ends-1)
        cp.cumsum(index_builder, out=index_builder)
        return index_builder[:-1], shape

    def _get_flat_indices_fast(self):
        shape = self.get_shape()
        index_builder = cp.ones(int(shape.size), dtype=self._dtype)
        index_builder[shape.starts[1:]] = cp.diff(self.starts) - self.lengths[:-1] + 1
        index_builder[0] = self.starts[0]
        cp.cumsum(index_builder, out=index_builder)
        shape.empty_removed = True
        return index_builder, shape

@dataclass
class CPRaggedView2(RaggedView2):
    def get_shape(self):
        return CPRaggedShape(cp.atleast_1d(self.lengths))

    def get_flat_indices(self, do_split=False):
        """Return the indices into a flattened array

        Return the indices of all the elements in all the
        rows in this view

        Returns
        -------
        array
        """
        if not self.n_rows:
            return cp.ones(0, dtype=self._dtype), self.get_shape()
        shape = self.get_shape()
        step = 1 if self.col_step is None else self.col_step
        index_builder = cp.full(int(shape.size) + 1, step, dtype=self._dtype)
        index_builder[shape.ends[::-1]] = 1 - self.ends[::-1]
        index_builder[0] = 0
        index_builder[shape.starts] += self.starts
        np.cumsum(index_builder, out=index_builder)
        return index_builder[:-1], shape

