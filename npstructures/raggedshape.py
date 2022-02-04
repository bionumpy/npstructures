from numbers import Number
import numpy as np

class ViewBase:
    _dtype = np.int32
    def __init__(self, codes, lengths=None):
        if lengths is None:
            self._codes = codes.view(self._dtype)
        else:
            starts = np.asanyarray(codes, dtype=self._dtype)
            lengths = np.asanyarray(lengths, dtype=self._dtype)
            if not lengths.size:
                self._codes = np.array([], dtype=self._dtype)
            else:
                self._codes = np.hstack((starts[:, None], lengths[:, None])).flatten()

    def __eq__(self, other):
       return np.all(self._codes==other._codes)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.starts}, {self.lengths})"

    @property
    def lengths(self):
        """The row lengths"""
        return self._codes[1::2]

    @property
    def starts(self):
        """The start index of each row"""
        return self._codes[::2]

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
        """Check wheter the `View` with certainty have no empty rows

        Returns
        -------
        bool
            Whether or not it is cerain that this view contins no empty rows
        """
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
        return self.starts[indices[0]]+np.asanyarray(indices[1], dtype=self._dtype)

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
        diffs = np.zeros(self.size, dtype=self._dtype)
        diffs[self.starts[1:]] = 1
        return np.cumsum(diffs)


class RaggedRow:
    _dtype=np.int32
    def __init__(self, code):
        code =  np.atleast_1d(code).view(self._dtype)
        self.starts = code[0]
        self.legths = code[1]
        self.ends = code[0]+code[1]

class RaggedShape(ViewBase):
    """ Class that represents the shape of a ragged array.
    
    Represents the same information as a list of row lengths.

    Parameters
    ----------
    codes : list or array_like
        Either a list of row lengths, or if ``is_coded=True`` an  array containing row-starts
        and row-lengths as 32-bit numbers.
    is_coded : bool, default=False
        if `False`, the `codes` are interpreted as row lengths.

    Attributes
    ----------
    starts
    lengths
    ends
    """
    def __init__(self, codes, is_coded=False):
        if is_coded: # isinstance(codes, np.ndarray) and codes.dtype==np.uint64:
            super().__init__(codes)
            self._is_coded = True
        else:
            lengths = np.asanyarray(codes, dtype=self._dtype)
            starts = np.insert(lengths.cumsum(dtype=self._dtype)[:-1], 0, self._dtype(0))
            super().__init__(starts, lengths)
            self._is_coded = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.lengths})"

    def __str__(self):
        return str(self.lengths)

    def __getitem__(self, index):
        if not isinstance(index, slice) or isinstance(index, Number):
            return NotImplemented
        if isinstance(index, Number):
            index = [index]
        new_codes = self._codes.view(np.uint64)[index].copy().view(self._dtype)
        new_codes[::2] -= new_codes[0]
        return self.__class__(new_codes, is_coded=True)

    @property
    def size(self):
        """The sum of the row lengths"""
        if not self.n_rows:
            return 0
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
        if isinstance(indices, Number):
            return RaggedRow(self._codes.view(np.uint64)[indices])
        return RaggedView(self._codes.view(np.uint64)[indices])

    def to_dict(self):
        """Return a `dict` of all necessary variables"""
        return {"codes": self._codes}

    @classmethod
    def from_dict(cls, d):
        """Load a `Shape` object from a dict of necessary variables

        Paramters
        ---------
        d : dict
            `dict` containing all the variables needed to initialize a RaggedShape

        Returns
        -------
        RaggedShape
        """
        if "offsets" in d:
            return cls(np.diff(d["offsets"]))
        else:
            return cls(d["codes"])

    @classmethod
    def asshape(cls, shape):
        """Create a `Shape` from either a list of row lengths or a `Shape`
        
        If `shape` is already a `RaggedShape`, do nothing. Else construct a new
        `RaggedShape` object

        Parameters
        ----------
        shape : RaggedShape or array_like

        Returns
        -------
        RaggedShape
        """
        if isinstance(shape, RaggedShape):
            return shape
        return cls(shape)

    def broadcast_values(self, values, dtype=None):
        """Broadcast the values in a column vector to the data of a ragged array

        The resulting array is such that a `RaggedArray` with `self` as shape will
        have the rows filled with the values in `values. I.e. 
        ``RaggedArray(ret, self)[row, j] = values[row, 1]``

        Parameters
        ----------
        values : array_like
            column vectors with values to be broadcasted
        
        Returns
        -------
        array
            flat array with broadcasted values
        """
        values = np.asanyarray(values)
        assert values.shape == (self.n_rows, 1), (values.shape, (self.n_rows, 1))
        if self.empty_rows_removed():
            return self._broadcast_values_fast(values, dtype)
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

    @classmethod
    def from_tuple_shape(cls, tuple_shape):
        assert len(tuple_shape) == 2, f"Can only converd 2d array: {tuple_shape}"
        lengths = np.full(tuple_shape[0], tuple_shape[1], dtype="int")
        return cls(lengths)

class RaggedView(ViewBase):
    """Class to represent a view onto subsets of rows

    Same as RaggedShape, except without the constraint that the rows 
    fill the whole data array. I.e. ``np.all(self.ends[:-1]==self.starts[1:])``
    does not necessarilty hold.

    Parameters
    ----------
    codes : array_like
        Either a list of row starts, or if `lengths` is provided an  array containing row-starts
        and row-lengths as 32-bit numbers.
    lengths : array_like, optional
        the lengths of the rows

    Attributes
    ----------
    starts
    lengths
    ends
    """

    def __getitem__(self, index):
        if isinstance(index, Number):
            return RaggedRow(self._codes.view(np.uint64)[index])

        return self.__class__(self._codes.view(np.uint64)[index])

    def get_shape(self):
        """ Return the shape of a ragged array containing the view's rows

        Returns
        -------
        RaggedShape
            The shape of a ragged array consisting of the rows in this view
        """
        if not self.n_rows:
            return RaggedShape(self._codes, is_coded=True)
        
        codes = self._codes.copy()
        np.cumsum(codes[1:-1:2], out=codes[2::2])
        codes[0] = 0
        return RaggedShape(codes, is_coded=True)

    def get_flat_indices(self):
        """Return the indices into a flattened array

        Return the indices of all the elements in all the
        rows in this view

        Returns
        -------
        array
        """
        if not self.n_rows:
            return np.ones(0, dtype=self._dtype), self.get_shape()

        if self.empty_rows_removed():
            return self._get_flat_indices_fast()
        shape = self.get_shape()
        index_builder = np.ones(shape.size+1, dtype=self._dtype)
        index_builder[shape.ends[::-1]] = 1-self.ends[::-1]
        index_builder[0] = 0
        index_builder[shape.starts] += self.starts
        np.cumsum(index_builder, out=index_builder)
        return index_builder[:-1], shape

    def _get_flat_indices_fast(self):
        shape = self.get_shape()
        index_builder = np.ones(shape.size, dtype=self._dtype)
        index_builder[shape.starts[1:]] = np.diff(self.starts)-self.lengths[:-1]+1
        index_builder[0] = self.starts[0]
        np.cumsum(index_builder, out=index_builder)
        shape.empty_removed = True
        return index_builder, shape
