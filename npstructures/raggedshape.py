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
        """The row lengths"""
        if isinstance(self._codes, Number):
            return np.atleast_1d(self._codes).view(np.int32)[1]
        return self._codes.view(np.int32)[1::2]

    @property
    def starts(self):
        """The start index of each row"""
        if isinstance(self._codes, Number):
            return np.atleast_1d(self._codes).view(np.int32)[0]

        return self._codes.view(np.int32)[::2]

    @property
    def ends(self):
        """The end index of each row"""
        return self.starts+self.lengths

    @property
    def n_rows(self):
        """Number of rows"""
        if isinstance(self.starts, Number):
            return 1
        return self.starts.size

    def empty_rows_removed(self):
        """Check wheter the `View` with certainty have no empty rows"""
        return hasattr(self, "empty_removed") and self.empty_removed
        
    def ravel_multi_index(self, indices):
        """Return the flattened indices of a set of array indices

        Parameters
        ----------
        indices : tuple
            Tuple containing the row- and column indices to ravel

        Returns
        -------
        array
            array containing the flattenened indices
        """
        return self.starts[indices[0]]+np.asanyarray(indices[1], dtype=np.int32)

    def unravel_multi_index(self, flat_indices):
        """Return array indices for a set of flat indices

        Parameters
        ----------
        indices : index_like
            flat indices to unravel

        Returns
        -------
        tuple
            tuple containing the unravelled row- and column indices
        """
        starts = self.starts
        rows = np.searchsorted(starts, flat_indices, side="right")-1
        cols = flat_indices-starts[rows]
        return rows, cols

    def index_array(self):
        """Return an array of broadcasted row indices"""
        diffs = np.zeros(self.size, dtype=np.int32)
        diffs[self.starts[1:]] = 1
        return np.cumsum(diffs)


class RaggedShape(ViewBase):
    """ Class that represents the shape of a ragged array.
    
    Represents the same information as a list of row lengths.

    Parameters
    ----------
    codes : list or array_like
        Either a list of row lengths, or an `uint64` array containing row-starts
        and row-lengths as 32-bit numbers.

    Notes
    -----
    The internal representation is a 64bit array containing both the row-start 
    and row-length in each register. This is done to make index-lookups faster, 
    since all information pertinent to a row is contained in a single row. 

    Since this uses the ``np.view`` functionality, files saved on one computer
    might not be correct when loaded on another computer, if they have different 
    endianness.
    """

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
        new_codes = self._codes[index].view(np.int32).copy()
        new_codes[::2] -= new_codes[0]
        return self.__class__(new_codes.view(np.uint64))

    @property
    def size(self):
        """The sum of the row lengths"""
        return self.starts[-1]+self.lengths[-1]

    def view(self, indices):
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
        return RaggedView(self._codes[indices])

    def to_dict(self):
        """Return a `dict` of all necessary variables"""
        return {"codes": self._codes}

    @classmethod
    def from_dict(cls, d):
        """Load a `Shape` object from a dict of necessary variables"""
        if "offsets" in d:
            return cls(np.diff(d["offsets"]))
        else:
            return cls(d["codes"])

    @classmethod
    def asshape(cls, shape):
        """Create a `Shape` from either a list of row lengths or a `Shape`"""
        if isinstance(shape, RaggedShape):
            return shape
        return cls(shape)

    def broadcast_values(self, values, dtype=None):
        values = np.asanyarray(values)
        assert values.shape == (self.n_rows, 1), (values.shape, (self.n_rows, 1))
        if self.empty_rows_removed():
            return self._broadcast_values_fast(values)
        values = values.ravel()
        broadcast_builder = np.zeros(self.size+1, dtype=dtype)
        broadcast_builder[self.ends[::-1]] -= values[::-1]
        broadcast_builder[0] = 0 
        broadcast_builder[self.starts] += values
        func = np.logical_xor if values.dtype==bool else np.add
        return func.accumulate(broadcast_builder[:-1])

    def _broadcast_values_fast(self, values, dtype=None):
        values = values.ravel()
        broadcast_builder = np.zeros(self.size, dtype=dtype)
        broadcast_builder[self.starts[1:]] = np.diff(values)
        broadcast_builder[0] = values[0]
        func = np.logical_xor if values.dtype==bool else np.add
        func.accumulate(broadcast_builder, out=broadcast_builder)
        return broadcast_builder

class RaggedView(ViewBase):
    """Class to represent a view onto subsets of rows
    """
    def __getitem__(self, index):
        return self.__class__(self._codes[index])

    def get_shape(self):
        """ Return the shape of a ragged array containing the view's rows"""
        codes = self._codes.copy().view(np.int32)
        np.cumsum(codes[1:-1:2], out=codes[2::2])
        codes[0] = 0 
        return RaggedShape(codes.view(np.uint64))

    def get_flat_indices(self):
        """Return the indices into a flattened array"""
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

