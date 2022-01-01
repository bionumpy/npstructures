from numbers import Number
import numpy as np

class ViewBase:
    def __init__(self, codes, lengths=None):
        if lengths is None:
            self._codes = codes
        else:
            starts = np.asanyarray(codes, dtype=np.int32)
            lengths = np.asanyarray(lengths, dtype=np.int32)
            self._codes = np.hstack((starts[:, None], lengths[:, None])).flatten().view(np.uint64)

    def __eq__(self, other):
       return np.all(self._codes==other._codes)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.starts}, {self.lengths})"

    @property
    def lengths(self):
        if isinstance(self._codes, Number):
            return np.atleast_1d(self._codes).view(np.int32)[1]
        return self._codes.view(np.int32)[1::2]

    @property
    def starts(self):
        if isinstance(self._codes, Number):
            return np.atleast_1d(self._codes).view(np.int32)[0]

        return self._codes.view(np.int32)[::2]

    @property
    def ends(self):
        return self.starts+self.lengths

    @property
    def n_rows(self):
        if isinstance(self.starts, Number):
            return 1
        return self.starts.size

    @classmethod
    def asshape(cls, shape):
        if isinstance(shape, RaggedShape):
            return shape
        return cls(shape)

    def empty_rows_removed(self):
        return hasattr(self, "empty_removed") and self.empty_removed
        
    def ravel_multi_index(self, indices):
        return self.starts[indices[0]]+np.asanyarray(indices[1], dtype=np.int32)

    def unravel_multi_index(self, flat_indices):
        starts = self.starts
        rows = np.searchsorted(starts, flat_indices, side="right")-1
        cols = flat_indices-starts[rows]
        return rows, cols

    def index_array(self):
        diffs = np.zeros(self.size, dtype=np.int32)
        diffs[self.starts[1:]] = 1
        return np.cumsum(diffs)


class RaggedShape(ViewBase):
    def __init__(self, codes):
        if isinstance(codes, np.ndarray) and codes.dtype==np.uint64:
            super().__init__(codes)            
        else:
            lengths = np.asanyarray(codes, dtype=np.int32)
            starts = np.insert(lengths.cumsum(dtype=np.int32)[:-1], 0, np.int32(0))
            super().__init__(starts, lengths)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.lengths})"

    def __str__(self):
        return str(self.lengths)

    def __getitem__(self, index):
        if not isinstance(index, slice) or isinstance(index, Number):
            return NotImplemented
        new_codes = self._codes[index].view(np.int32)
        new_codes[::2] -= new_codes[0]
        return self.__class__(new_codes.view(np.uint64))

    @property
    def size(self):
        return self.starts[-1]+self.lengths[-1]

    def view(self, indices):
        return RaggedView(self._codes[indices])

    def to_dict(self):
        return {"codes": self._codes}

    @classmethod
    def from_dict(cls, d):
       if "offsets" in d:
            return cls(np.diff(d["offsets"]))
       else:
          return cls(d["codes"])

class RaggedView(ViewBase):
    def __getitem__(self, index):
        return self.__class__(self._codes[index])

    def get_shape(self):
        codes = self._codes.copy().view(np.int32)
        np.cumsum(codes[1:-1:2], out=codes[2::2])
        codes[0] = 0 
        return RaggedShape(codes.view(np.uint64))

    def get_flat_indices(self):
        if self.empty_rows_removed():
            return self._get_flat_indices_fast()
        shape = self.get_shape()
        index_builder = np.ones(shape.size+1, dtype=np.int32)
        index_builder[shape.ends[::-1]] = 1-self.ends[::-1]
        index_builder[0] = 0
        index_builder[shape.starts] += self.starts
        np.cumsum(index_builder, out=index_builder)
        return index_builder[:-1], shape

    def _get_flat_indices_fast(self):
        shape = self.get_shape()
        index_builder = np.ones(shape.size, dtype=np.int32)
        index_builder[shape.starts[1:]] = np.diff(self.starts)-self.lengths[:-1]+1
        index_builder[0] = shape.starts[0]
        np.cumsum(index_builder, out=index_builder)
        shape.empty_removed = True
        return index_builder, shape
