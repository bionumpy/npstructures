from numbers import Number
from dataclasses import dataclass
#try:
#    from .copy_segment import compute
#except ImportError:
compute = None
from .util import np

def simple_build_indices(view, to_shape, step):
    lengths = to_shape.lengths
    diffs = np.arange(int(to_shape.size)+1, dtype=view._dtype)
    if step is not None and step != 1:
        diffs *= step
    change_indices = np.insert(np.cumsum(lengths)[:-1], 0, 0)
    diffs[:-1] += np.repeat(view.starts - diffs[change_indices], lengths)
    return diffs[:-1], to_shape


def build_indices(view, to_shape, step):
    # return simple_build_indices(view, to_shape, step)
    step = 1 if step is None else step
    if to_shape.size == 0:
        return np.zeros(0, dtype=view._dtype), to_shape
    index_builder = np.full(int(to_shape.size + 1), step, dtype=view._dtype)
    #empty_rows = np.flatnonzero(to_shape.lengths == 0)
    non_empty_mask = to_shape.lengths != 0
    if not np.all(non_empty_mask):
        idx = np.flatnonzero(non_empty_mask)
        #shape_starts = to_shape.starts[row_indices]
        #view_starts = view.starts[row_indices]
        #view_ends = view.ends[row_indices]
    else:
        idx = slice(None)
        #shape_starts = to_shape.starts
        # view_starts = view.starts
        #view_ends = view.ends

    index_builder[to_shape.starts[idx][1:]] = view.starts[idx][1:]-view.ends[idx][:-1] + 1
    index_builder[0] = view.starts[idx[0] if not isinstance(idx, slice) else 0]
    np.cumsum(index_builder, out=index_builder)
    return index_builder[:-1], to_shape


@dataclass
class BasicView:
    starts: np.ndarray
    ends: np.ndarray
    _dtype: np.dtype = np.int64


def native_extract_segments(input_array, view, to_shape, step):
    return input_array[build_indices(view, to_shape, step)[0]]


def c_extract_segments(input_array, view, to_shape, step):
    if compute is None:
        return native_extract_segments(input_array, view, to_shape, step)
    new_array = np.empty_like(input_array, shape=(to_shape.size,))
    compute(input_array, new_array, view.starts, view.ends, step)
    return new_array


class ViewBase:
    _dtype = np.int64

    @classmethod
    def set_dtype(cls, dtype):
        cls._dtype = dtype

    def __init__(self, codes, lengths=None, step=None):
        if lengths is None:
            self._codes = codes.view(self._dtype)
        else:
            starts = np.asanyarray(codes, dtype=self._dtype)
            lengths = np.asanyarray(lengths, dtype=self._dtype)
            if not lengths.size:
                self._codes = np.array([], dtype=self._dtype)
            else:
                self._codes = np.hstack((starts[:, None], lengths[:, None])).flatten()
        self._step = step

    def __eq__(self, other):
        return isinstance(other, ViewBase) and np.all(self._codes == other._codes)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.starts}, {self.lengths}, {self._step})"

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
        return self.starts + self.lengths

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
        return self.starts[indices[0]] + np.asanyarray(indices[1], dtype=self._dtype)

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
        rows = np.searchsorted(starts, flat_indices, side="right") - 1
        cols = flat_indices - starts[rows]
        return rows, cols

    def index_array(self):
        """Return an array of broadcasted row indices"""
        # diffs = np.zeros(self.size + 1, dtype=self._dtype)
        diffs = np.bincount(self.starts[1:], minlength=self.size + 1)
        # diffs[self.starts[1:]] = 1
        return np.cumsum(diffs)[:-1]

    def _index_rows(self, idx):
        if self._dtype == np.int32:
            return np.atleast_1d(self._codes.view(np.uint64)[idx]).view(self._dtype)
        else:
            return self._codes.reshape(-1, 2)[idx].ravel()

    def view_cols(self, idx):
        if isinstance(idx, Number):
            if idx >= 0:
                return RaggedView(self.starts + idx, np.ones_like(self.lengths))
            return RaggedView(self.ends + idx, np.ones_like(self.lengths))
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
                starts = starts + np.minimum(lengths, col_slice.start)
            else:
                starts = starts + np.maximum(lengths + col_slice.start, 0)
        if col_slice.stop is not None:
            if col_slice.stop >= 0:
                ends = np.minimum(self.starts + col_slice.stop, ends)
            else:
                ends = np.maximum(self.ends + col_slice.stop, starts)

        return RaggedView(starts, np.maximum(0, ends - starts), step=col_slice.step)


class RaggedRow:
    def __init__(self, code):
        self.starts = code[0]
        self.lengths = code[1]
        self.ends = code[0] + code[1]

    def view_cols(self, idx):
        if isinstance(idx, Number):
            if idx >= 0:
                return RaggedView(self.starts + idx, np.ones_like(self.lengths))
            return RaggedView(self.ends + idx, np.ones_like(self.lengths))
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
                starts = starts + np.minimum(lengths, col_slice.start)
            else:
                starts = starts + np.maximum(lengths + col_slice.start, 0)
        if col_slice.stop is not None:
            if col_slice.stop >= 0:
                ends = np.minimum(self.starts + col_slice.stop, ends)
            else:
                ends = np.maximum(self.ends + col_slice.stop, starts)

        return RaggedView(np.array([starts]), np.array([np.maximum(0, ends - starts)]), col_slice.step)


class RaggedShape(ViewBase):
    """Class that represents the shape of a ragged array.

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
        if is_coded:
            super().__init__(codes)
            self._is_coded = True
        else:
            lengths = np.asanyarray(codes, dtype=self._dtype)
            starts = np.pad(np.cumsum(lengths, dtype=self._dtype)[:-1], pad_width=1, mode="constant")[:-1]

            super().__init__(starts, lengths)
            self._is_coded = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.lengths})"

    def __str__(self):
        return repr(self) #str(self.lengths)

    def __getitem__(self, index):
        if not isinstance(index, slice) or isinstance(index, Number):
            return NotImplemented
        if isinstance(index, Number):
            index = [index]
        new_codes = self._index_rows(index).copy()
        # new_codes = self._codes.view(np.uint64)[index].copy().view(self._dtype)
        new_codes[::2] -= new_codes[0]
        return self.__class__(new_codes, is_coded=True)

    @property
    def size(self):
        """The sum of the row lengths"""
        if not self.n_rows:
            return 0
        return self.starts[-1] + self.lengths[-1]

    def view_rows(self, indices):
        idx = self._index_rows(indices).reshape(-1, 2)
        return RaggedView2(idx[..., 0], idx[..., 1])

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
            return RaggedRow(self._index_rows(indices))
        # self._codes.view(np.uint64)[indices])
        # return RaggedView(self._codes.view(np.uint64)[indices])
        return RaggedView(self._index_rows(indices))

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
            return cls(d["codes"], is_coded=True)

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
        if isinstance(shape, tuple) and len(shape) == 2:
            return cls(shape[-1])
        return cls(shape)

    def _get_accumulation_func(self, dtype):
        return np.logical_xor.accumulate if dtype == bool else np.add.accumulate

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
        values = np.asanyarray(values, dtype=dtype)
        if values.size == 1:
            return values.ravel()
        assert values.shape == (self.n_rows, 1), (values.shape, (self.n_rows, 1))
        values = values.ravel()
        if self.empty_rows_removed():
            return self._broadcast_values_fast(values, dtype)
        return self._raw_broadcast(values, dtype)

    def _raw_broadcast(self, values, dtype=None):
        dtypes = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64, 16: "uint128"}
        orig_dtype = values.dtype
        values = values.view(dtypes[orig_dtype.itemsize])
        broadcast_builder = np.zeros(self.size + 1, dtype=values.dtype)
        broadcast_builder[self.ends[::-1]] ^= values[::-1]
        broadcast_builder[0] = 0
        broadcast_builder[self.starts] ^= values
        accumulation_func = np.bitwise_xor.accumulate
        return accumulation_func(broadcast_builder[:-1]).view(orig_dtype)

    def _broadcast_values_fast(self, values, dtype=None):
        values = values.ravel()
        broadcast_builder = np.zeros(self.size, dtype=dtype)
        broadcast_builder[self.starts[1:]] = np.diff(values)
        broadcast_builder[0] = values[0]
        accumulation_func = self._get_accumulation_func(values.dtype)
        accumulation_func(broadcast_builder, out=broadcast_builder)
        return broadcast_builder

    @classmethod
    def from_tuple_shape(cls, tuple_shape):
        assert len(tuple_shape) == 2, f"Can only convert 2d array: {tuple_shape}"
        lengths = np.full(tuple_shape[0], tuple_shape[1], dtype="int")
        return cls(lengths)


@dataclass
class RaggedView2:
    starts: np.ndarray
    lengths: np.ndarray
    col_step: int = 1
    _dtype: int = np.int64

    def __post_init__(self):
        self.starts = np.atleast_1d(self.starts)
        self.lengths = np.atleast_1d(self.lengths)

    def view_rows(self, indices):
        return self.__class__(self.starts[indices],
                              self.lengths[indices],
                              self.col_step,
                              self._dtype)

    def view(self, indices):
        return self.view_rows(indices)

    @property
    def n_rows(self):
        if isinstance(self.starts, Number):
            return 1
        return len(self.starts)

    def row_slice(self, row_slice):
        return self.__class__(self.starts[row_slice], self.ends[row_slice], self.col_step)

    def _calculate_lengths(self, col_slice):
        """ Figure out how long each line will be after a col slice"""

        start, stop, step = (col_slice.start, col_slice.stop, col_slice.step)
        if step is None:
            step = 1
        assert step != 0
        if start is None:
            start = 0 if step >= 0 else self.lengths-1
        elif start < 0:
            start = self.lengths+start
        if stop is None:
            stop = self.lengths if step >= 0 else -1
        elif stop < 0:
            stop = self.lengths+stop

        mask = np.sign(stop-start) != np.sign(step) # start is higher than stop and step is not negative
        mask |= (start < 0) & (step < 0) # start is before 0 and step is negative
        mask |= (start >= self.lengths) & (step > 0) # start is  after end and step is negative
        mask |= (stop <= 0) & (step > 0) # stop is before 0 and step is positive
        mask |= (stop >= self.lengths) & (step < 0) # stop is after end and step is negative
        start = np.maximum(np.minimum(start, self.lengths-1),
                           0) #put start in range
        d = 0 if step >= 0 else -1
        stop = np.maximum(np.minimum(stop, self.lengths+d),
                          0+d) # put stop in range
        L = stop-start #length
        # mask = np.sign(L) != np.sign(step)
        return np.where(mask, 0,
                        (np.abs(L)-1)//np.abs(step)+1)

    def _pos_col_slice(self, col_slice):
        assert col_slice.step > 0
        start = col_slice.start
        if start is None:
            start = 0
        elif start >= 0:
            start = np.minimum(start, self.lengths)
            # new_start = self.starts + np.minimum(start, self.lengths)*self.col_step
        else:
            start = np.maximum(self.lengths+start, 0)
            # new_start = self.starts + np.maximum(self.lengths+start, 0)*self.col_step
        stop = col_slice.stop
        if stop is None:
            stop = self.lengths
        elif stop < 0:
            stop = np.maximum(self.lengths + stop, 0)
        else:
            stop = np.minimum(self.lengths, stop)
        return self.__class__(self.starts+self.col_step*start,
                              np.maximum(0, (stop-start+(col_slice.step-1))//col_slice.step),
                              self.col_step*col_slice.step)

    def col_slice(self, col_slice):
        if isinstance(col_slice, Number):
            idx = col_slice
            if idx >= 0:
                return self.__class__(self.starts + idx,
                                      np.ones_like(self.lengths))
            return self.__class__(self.ends + idx, np.ones_like(self.lengths))

        # starts, lengths, col_step = (self.starts, self.lengths, self.col_step)
        step = 1 if col_slice.step is None else col_slice.step
        if step > 0:
            return self._pos_col_slice(slice(col_slice.start, col_slice.stop, step))
        col_slice_start = col_slice.start
        if col_slice_start is None:
            col_slice_start = 0 if step >= 0 else self.lengths-1
        elif col_slice_start < 0:
            col_slice_start = self.lengths + col_slice_start
        col_slice_start = np.maximum(
            np.minimum(self.lengths-1, col_slice_start),
            0)

        lengths = self._calculate_lengths(col_slice)
        starts = self.starts + self.col_step*col_slice_start
        return self.__class__(starts, lengths, step*self.col_step)

    def get_shape(self):
        return RaggedShape(np.atleast_1d(self.lengths))

    @property
    def ends(self):
        return self.starts + (self.lengths-1)*self.col_step+1

    def _get_flat_indices(self, do_split=False):
        """Return the indices into a flattened array

        Return the indices of all the elements in all the
        rows in this view

        Returns
        -------
        array
        """
        if not self.n_rows:
            return np.ones(0, dtype=self._dtype), self.get_shape()
        shape = self.get_shape()
        return build_indices(self, shape, self.col_step)
        # step = 1 if self.col_step is None else self.col_step
        # index_builder = np.full(shape.size + 1, step, dtype=self._dtype)
        # np.add.at(index_builder, shape.starts[1:], self.starts[1:]-self.ends[:-1]-step+1)
        # np.add.at(index_builder, 0, self.starts[0]-step)
        # np.cumsum(index_builder, out=index_builder)
        # return index_builder[:-1], shape

    def get_flat_indices(self, do_split=False):
        return self._get_flat_indices(do_split)
        if self.col_step != 1 or np.any(self.starts[:-1] > self.starts[1:]):
            return self._get_flat_indices(do_split)
        if not self.n_rows:
            return np.ones(0, dtype=self._dtype), self.get_shape()
        mask_builder = np.zeros(self.ends[-1]+1, dtype=bool)
        np.bitwise_xor.at(mask_builder, self.starts, True)
        np.bitwise_xor.at(mask_builder, self.ends, True)
        np.bitwise_xor.accumulate(mask_builder, out=mask_builder)
        return mask_builder[:-1], self.get_shape()


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
            return RaggedRow(
                self._index_rows(index)
            )  # self._codes.view(np.uint64)[index])

        return self.__class__(
            self._index_rows(index)
        )  # self._codes.view(np.uint64)[index])

    def view(self, indices):
        return self.__class__(self._index_rows(indices))

    def view_rows(self, indices):
        idx = self._index_rows(indices).reshape(-1, 2)
        return RaggedView2(idx[..., 0], idx[..., 1])

    def get_shape(self):
        """Return the shape of a ragged array containing the view's rows

        Returns
        -------
        RaggedShape
            The shape of a ragged array consisting of the rows in this view
        """
        if not self.n_rows:
            return RaggedShape(self._codes, is_coded=True)

        codes = self._codes.copy()
        if self._step is not None:
            codes[1::2] //= np.abs(self._step)
        np.cumsum(codes[1:-1:2], out=codes[2::2])
        codes[0] = 0
        return RaggedShape(codes, is_coded=True)

    def _build_indices(self, shape):
        step = 1 if self._step is None else self._step
        index_builder = np.full(shape.size + 1, step, dtype=self._dtype)
        func = np.add if step >= 0 else np.subtract
        func.at(index_builder, shape.starts[1:], self.starts[1:]-self.ends[:-1])
        func.at(index_builder, 0, self.starts[0]-1)
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
            return np.ones(0, dtype=self._dtype), self.get_shape()

        if self.empty_rows_removed():
            return self._get_flat_indices_fast()
        shape = self.get_shape()
        chunk_size = 100000
        if do_split and self.starts.size > chunk_size:
            slices = (slice(i*chunk_size, (i+1)*chunk_size) for i in range((len(self.starts)-1)//chunk_size+1))
            return (self[s]._build_indices(shape[s])[0] for s in slices), shape
        return build_indices(self, shape, self._step)
        return self._build_indices(shape)

    def _get_flat_indices_fast(self):
        shape = self.get_shape()
        index_builder = np.ones(shape.size, dtype=self._dtype)
        index_builder[shape.starts[1:]] = np.diff(self.starts) - self.lengths[:-1] + 1
        index_builder[0] = self.starts[0]
        np.cumsum(index_builder, out=index_builder)
        shape.empty_removed = True
        return index_builder, shape
