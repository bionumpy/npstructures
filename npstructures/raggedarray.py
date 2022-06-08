import numpy as np
from numbers import Number
from .raggedshape import RaggedShape, RaggedView
from .arrayfunctions import HANDLED_FUNCTIONS, REDUCTIONS, ACCUMULATIONS


def row_reduction(func):
    def new_func(self, axis=None, keepdims=False):
        if axis is None:
            return getattr(np, func.__name__)(self._data)
        if axis in (1, -1):
            r = func(self)
            if keepdims:
                r = r[:, None]
            return r
        return NotImplemented

    return new_func


INVERSE_FUNCS = {
    np.add: (np.subtract, np.add),
    np.subtract: (np.subtract, np.add),
    np.bitwise_xor: (np.bitwise_xor, np.bitwise_xor),
}


class RaggedArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Class to represent 2d arrays with differing row lengths

    Provides objects that behaves similar to numpy ndarrays, but
    can represent arrays with differing row lengths. Numpy-like functionality is
    provided in three ways.

    #. ufuncs are supported, so addition, multiplication, sin etc. works just like numpy arrays
    #. Indexing works similar to numpy arrays. Slice objects, index lists and boolean arrays etc.
    #. Some numpy functions like `np.concatenate`, `np.sum`, `np.nonzero` are implemented.

    See the examples for simple demonstrations.

    Parameters
    ----------
    data : nested list or array_like
        the nested list to be converted, or (if `shape` is provided) a continous array of values
    shape : list or `Shape`, optional
        the shape of the ragged array, or list of row lenghts
    dtype : optional
        the data type to use for the array

    Attributes
    ----------
    shape : RaggedShape
        the shape (row-lengths) of the array
    size : int
        the total size of the array
    dtype
        the data-type of the array

    Examples
    --------
    >>> ra = RaggedArray([[2, 4, 8], [3, 2], [5, 7, 3, 1]])
    >>> ra+1
    RaggedArray([[3, 5, 9], [4, 3], [6, 8, 4, 2]])
    >>> ra*2
    RaggedArray([[4, 8, 16], [6, 4], [10, 14, 6, 2]])
    >>> ra[1]
    array([3, 2])
    >>> ra[0:2]
    RaggedArray([[2, 4, 8], [3, 2]])
    >>> np.nonzero(ra>3)
    (array([0, 0, 2, 2]), array([1, 2, 0, 1]))
    """

    def __init__(self, data, shape=None, dtype=None, safe_mode=True):
        if shape is None:
            data, shape = self._from_array_list(data, dtype)
        elif isinstance(shape, RaggedShape):
            shape = shape
        else:
            shape = RaggedShape.asshape(shape)

        self.shape = shape
        self._data = np.asanyarray(data, dtype=dtype)
        self.size = self._data.size
        self.dtype = self._data.dtype
        self._safe_mode = safe_mode

    def astype(self, dtype):
        return RaggedArray(self._data.astype(dtype), self.shape)

    def __len__(self):
        return self.shape.n_rows

    def __iter__(self):
        return (
            self._data[start : start + l]
            for start, l in zip(self.shape.starts, self.shape.lengths)
        )

    def __repr__(self):
        if len(self) < 20:
            return f"{self.__class__.__name__}({self.tolist()})"
        return f"{self.__class__.__name__}({self._data}, {self.shape})"

    def __str__(self):
        if len(self) < 20:
            return str(self.tolist())
        return f"{self.__class__.__name__}({self._data}, {self.shape})"

    def __iadd__(self, other):
        if isinstance(other, Number):
            self._data += other
        elif isinstance(other, RaggedArray):
            assert self.shape == other.shape
            self._data += other._data
        else:
            return NotImplemented

        return self

    def save(self, filename):
        """Saves the ragged array to file using np.savez

        Parameters
        ----------
        filename : str
            name of the file
        """
        np.savez(filename, data=self._data, **(self.shape.to_dict()))

    @classmethod
    def load(cls, filename):
        """Loads a ragged array from file

        Parameters
        ----------
        filename : str
            name of the file

        Returns
        -------
        RaggedArray
            The ragged array loaded from file
        """
        D = np.load(filename)
        shape = RaggedShape.from_dict(D)
        return cls(D["data"], shape)

    def equals(self, other):
        """Checks for euqality with `other`"""
        return self.shape == other.shape and np.all(self._data == other._data)

    def tolist(self):
        """Returns a list of list of rows"""
        return [row.tolist() for row in self]

    def to_numpy_array(self):
        if len(self) == 0:
            return np.empty(shape=(0, 0))
        L = self.shape.lengths[0]
        assert np.all(self.shape.lengths == L)
        return self._data.reshape(self.shape.n_rows, L)

    @classmethod
    def from_numpy_array(cls, array):
        shape = RaggedShape.from_tuple_shape(array.shape)
        return cls(array.ravel(), shape)

    @classmethod
    def _from_array_list(cls, array_list, dtype=None):
        data = np.array(
            [element for array in array_list for element in array], dtype=dtype
        )  # This can be done faster
        return data, RaggedShape([len(a) for a in array_list])

    ########### Indexing
    def __getitem__(self, index):
        ret = self._get_row_subset(index)
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
        view = self.shape.view(rows)
        view = view.view_cols(cols)
        ret, shape = self._get_view(view)
        if isinstance(rows, Number) or isinstance(cols, Number):
            shape = None
        return ret, shape
    # @example((np.array([[0]], dtype=np.int16), (slice(None, None, None), 0)))

    def _get_row_subset(self, index):
        if isinstance(index, tuple):
            if len(index) == 0:
                return slice(None), self.shape
            if len(index) == 1:
                index = index[0]
            else:
                assert len(index) == 2
                return self._get_row_col_subset(index[0], index[1])
        if index is Ellipsis:
            return slice(None), self.shape
        elif isinstance(index, Number):
            return self._get_row(index)
        elif isinstance(index, slice):
            if not ((index.step is None) or index.step == 1):
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
            elif isinstance(value, RaggedArray):
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
        flat_idx = self.shape.starts[row] + col
        return flat_idx, None

    def _get_rows(self, from_row, to_row):
        if from_row is None:
            from_row = 0
        if to_row is None:
            to_row = len(self)
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

    ### Broadcasting
    def _broadcast_rows(self, values):
        data = self.shape.broadcast_values(values, dtype=self.dtype)
        return self.__class__(data, self.shape)

    def _reduce(self, ufunc, ra, axis=0, **kwargs):
        assert axis in (
            1,
            -1,
        ), "Reductions on ragged arrays are only supported for the last axis"
        if ufunc not in REDUCTIONS:
            return NotImplemented
        return getattr(np, REDUCTIONS[ufunc])(ra, axis=axis, **kwargs)

    def _accumulate(self, ufunc, ra, axis=0, **kwargs):
        if ufunc in (np.add, np.subtract, np.bitwise_xor):
            return self._row_accumulate(ufunc)
        if ufunc not in ACCUMULATIONS:
            return NotImplemented
        return getattr(np, ACCUMULATIONS[ufunc])(ra, axis=axis, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method not in ("__call__", "reduce", "accumulate"):
            return NotImplemented
        if method == "reduce":
            return self._reduce(ufunc, inputs[0], **kwargs)
        if method == "accumulate":
            return self._accumulate(ufunc, inputs[0], **kwargs)
        datas = []
        for input in inputs:
            if isinstance(input, Number):
                datas.append(input)
            elif isinstance(input, np.ndarray) or isinstance(input, list):
                broadcasted = self._broadcast_rows(input)
                datas.append(broadcasted._data)
            elif isinstance(input, self.__class__):
                datas.append(input._data)
                if self._safe_mode and (input.shape != self.shape):
                    raise TypeError("inconsistent sizes")
            else:
                return NotImplemented
        return self.__class__(ufunc(*datas, **kwargs), self.shape)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def fill(self, value):
        self._data.fill(value)

    def ravel(self):
        """Return a contiguous flattened array.

        No copy is made of the data. Same as a concatenation of
        all the rows in the ragged array"""

        return self._data

    def nonzero(self):
        """Return the row- and column indices of nonzero elements"""
        flat_indices = np.flatnonzero(self._data)
        return self.shape.unravel_multi_index(flat_indices)

    # Reductions
    @row_reduction
    def sum(self):
        """Calculate sum or rowsums of the array

        Parameters
        ----------
        axis : int, optional
            If `None` compute sum for whole array. If `-1` compute row sums
        keepdims : bool, default=False
            If `True` return a column vector for row sums

        Returns
        -------
        int or array_like
            If `axis` is None, the sum of the whole array. If ``axis in (1, -1)``
            array containing the row sums
        """
        return np.bincount(
            self.shape.index_array(), self._data, minlength=self.shape.starts.size
        )

    @row_reduction
    def prod(self):
        return NotImplemented

    @row_reduction
    def mean(self):
        """Calculate mean or row means of the array

        Parameters
        ----------
        axis : int, optional
            If `None` compute mean of whole array. If `-1` compute row means
        keepdims : bool, default=False
            If `True` return a column vector for row sums

        Returns
        -------
        int or array_like
            If `axis` is None, the mean of the whole array. If ``axis in (1, -1)``
            array containing the row means
        """
        return self.sum(axis=-1) / self.shape.lengths

    @row_reduction
    def std(self):
        """Calculate standard deviation or row-std of the array

        Parameters
        ----------
        axis : int, optional
            If `None` compute standard deviation of whole array. If `-1` compute row std
        keepdims : bool, default=False
            If `True` return a column vector for row std

        Returns
        -------
        int or array_like
            If `axis` is None, the std of the whole array. If ``axis in (1, -1)``
            array containing the row stds
        """
        K = np.mean(self._data)
        a = ((self - K) ** 2).sum(axis=-1)
        b = (self - K).sum(axis=-1) ** 2
        return np.sqrt((a - b / self.shape.lengths) / self.shape.lengths)

    @row_reduction
    def all(self):
        """Check if all elements of the array are ``True``

        Returns
        -------
        bool
            Whether or not all elements evaluate to ``True``
        """
        true_counts = np.insert(np.cumsum(self._data), 0, 0)
        return (
            true_counts[self.shape.ends] - true_counts[self.shape.starts]
            == self.shape.lengths
        )

    @row_reduction
    def any(self):
        """Check if any elements of the array are ``True``

        Returns
        -------
        bool
            Whether or not all elements evaluate to ``True``
        """
        true_counts = np.insert(np.cumsum(self._data), 0, 0)
        return true_counts[self.shape.ends] - true_counts[self.shape.starts] > 0

    @row_reduction
    def max(self):
        assert np.all(self.shape.lengths)
        m = np.max(np.abs(self._data))
        offsets = 2 * m * np.arange(self.shape.n_rows)
        with_offsets = self + offsets[:, None]
        data = np.maximum.accumulate(with_offsets._data)
        return data[self.shape.ends - 1] - offsets

    @row_reduction
    def min(self):
        return -(-self).max(axis=-1)

    @row_reduction
    def argmax(self):
        m = self.max(axis=-1, keepdims=True)
        rows, cols = np.nonzero(self == m)
        _, idxs = np.unique(rows, return_index=True)
        return cols[idxs]

    @row_reduction
    def argmin(self, axis=None):
        return (-self).argmax(axis=-1)

    def cumsum(self, axis=None, dtype=None):
        if axis is None:
            return self._data.cumsum(dtype=dtype)
        if axis in (1, -1):
            cm = self._data.cumsum(dtype=dtype)
            offsets = np.insert(cm[self.shape.starts[1:] - 1], 0, 0)
            ra = self.__class__(cm, self.shape)
            return ra - offsets[:, None]

    def _row_accumulate(self, operator, dtype=None):
        starts = self._data[self.shape.starts]
        cm = operator.accumulate(self._data, dtype=dtype)
        offsets = INVERSE_FUNCS[operator][0](
            starts, cm[self.shape.starts]
        )  # TODO: This is the inverse
        ra = self.__class__(cm, self.shape)
        return INVERSE_FUNCS[operator][1](ra, offsets[:, None])

    def cumprod(self, axis=None, dtype=None):
        if axis is None:
            return self._data.cumprod(dtype=dtype)
        if axis in (1, -1):
            return NotImplemented

    def sort(self, axis=-1):
        if axis is None:
            return self._data.sort()
        if axis in (1, -1):
            args = np.lexsort((self._data, self.shape.index_array()))
            return self.__class__(self._data[args], self.shape)
