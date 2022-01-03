import numpy as np
from numbers import Number
from itertools import chain
from .raggedshape import RaggedShape, RaggedView
from .arrayfunctions import HANDLED_FUNCTIONS

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

    def __init__(self, data, shape=None, dtype=None):
        if shape is None:
            data, shape = self._from_array_list(data, dtype)
        else:
            shape = RaggedShape.asshape(shape)
        self.shape = shape
        self._data = np.asanyarray(data)
        self.size = self._data.size
        self.dtype = self._data.dtype

    def __len__(self):
        return self.shape.n_rows

    def __iter__(self):
        return (self._data[start:start+l] for start, l in zip(self.shape.starts, self.shape.lengths))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.tolist()})"

    def __str__(self):
        return str(self.tolist())

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
        """Checks for euqality with `other` """
        return self.shape == other.shape and np.all(self._data==other._data)

    def tolist(self):
        """Returns a list of list of rows"""
        return [row.tolist() for row in self]

    @classmethod
    def _from_array_list(cls, array_list, dtype=None):
        data = np.array([element for array in array_list for element in array], dtype=dtype) # This can be done faster
        return data, RaggedShape([len(a) for a in array_list])

    ########### Indexing
    def __getitem__(self, index):
        if isinstance(index, tuple):
            assert len(index)==2
            return self._get_element(index[0], index[1])
        elif isinstance(index, Number):
            return self._get_row(index)
        elif isinstance(index, slice):
            assert (index.step is None) or index.step==1
            return self._get_rows(index.start, index.stop)
        elif isinstance(index, np.ndarray) and index.dtype==bool:
            return self._get_rows_from_boolean(index)
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return self._get_multiple_rows(index)
        elif isinstance(index, RaggedView):
            return self._get_view(index)
        else:
            return NotImplemented

    def _get_row(self, index):
        assert 0 <= index < self.shape.n_rows, (0, index, self.shape.n_rows)
        view = self.shape.view(index)
        return self._data[view.starts:view.ends]

    def _get_rows(self, from_row, to_row):
        data_start = self.shape.view(from_row).starts
        new_shape = self.shape[from_row:to_row]
        data_end = data_start+new_shape.size
        new_data = self._data[data_start:data_end]
        return self.__class__(new_data, new_shape)

    def _get_rows_from_boolean(self, boolean_array):
        rows = np.flatnonzero(boolean_array)
        return self._get_multiple_rows(rows)

    def _get_view(self, view):
        indices, shape = view.get_flat_indices()
        new_data = self._data[indices]
        return self.__class__(new_data, shape)

    def _get_multiple_rows(self, rows):
        return self._get_view(self.shape.view(rows))

    def _get_element(self, row, col):
        flat_idx = self.shape.starts[row] + col
        assert np.all(flat_idx < self.shape.ends[row])
        return self._data[flat_idx]

    ### Broadcasting
    def _broadcast_rows(self, values):
        if self.shape.empty_rows_removed():
            return self._broadcast_rows_fast(values)
        assert values.shape == (self.shape.n_rows, 1)
        values = values.ravel()
        broadcast_builder = np.zeros(self._data.size+1, self.dtype)
        broadcast_builder[self.shape.ends[::-1]] -= values[::-1]
        broadcast_builder[0] = 0 
        broadcast_builder[self.shape.starts] += values
        func = np.logical_xor if values.dtype==bool else np.add
        return self.__class__(func.accumulate(broadcast_builder[:-1]), self.shape)

    def _broadcast_rows_fast(self, values):
        values = values.ravel()
        broadcast_builder = np.zeros(self._data.size, self.dtype)
        broadcast_builder[self.shape.starts[1:]] = np.diff(values)
        broadcast_builder[0] = values[0]
        func = np.logical_xor if values.dtype==bool else np.add
        func.accumulate(broadcast_builder, out=broadcast_builder)
        return self.__class__(broadcast_builder, self.shape)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
        
        datas = []
        for input in inputs:
            if isinstance(input, Number):
                datas.append(input)
            elif isinstance(input, np.ndarray):
                broadcasted = self._broadcast_rows(input)
                datas.append(broadcasted._data)
            elif isinstance(input, self.__class__):
                datas.append(input._data)
                # if np.any(input.shape != self.shape):
                #     raise TypeError("inconsistent sizes")
            else:
                return NotImplemented
        return self.__class__(ufunc(*datas, **kwargs), self.shape)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def ravel(self):
        """Return a contiguous flattened array.

        No copy is made of the data. Same as a concatenation of
        all the rows in the ragged array"""

        return self._data

    def nonzero(self):
        """Return the row- and column indices of nonzero elements"""
        flat_indices = np.flatnonzero(self._data)
        return self.shape.unravel_multi_index(flat_indices)

    def sum(self, axis=None):
        """Calculate sum or rowsums of the array

        Parameters
        ----------
        axis : int, optional
            If `None` compute sum for whole array. If `-1` compute row sums

        Returns
        -------
        int or array_like
            If `axis` is None, the sum of the whole array. If ``axis in (1, -1)`` 
            array containing the row sums
        """
        
        if axis is None:
            return self._data.sum()
        if axis == -1 or axis==1:
            return np.bincount(self.shape.index_array(), self._data, minlength=self.shape.starts.size)
        return NotImplemented

    def mean(self, axis=None):
        """ Calculate mean or row means of the array

        Parameters
        ----------
        axis : int, optional
            If `None` compute mean of whole array. If `-1` compute row means

        Returns
        -------
        int or array_like
            If `axis` is None, the mean of the whole array. If ``axis in (1, -1)`` 
            array containing the row means
        """

        s = self.sum(axis=axis)
        if axis is None:
            return s/self._data.size
        if axis == -1 or axis==1:
            return s/self.shape.lengths
        return NotImplemented

    def all(self, axis=None):
        """ Check if all elements of the array are True

        Returns
        -------
        bool
            Wheter or not all elements evaluate to ``True``
        """
        if axis is None:
            return np.all(self._data)
        return NotImplemented
