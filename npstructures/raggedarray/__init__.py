import types
from typing import Tuple, Dict, List, Union
import numpy.typing as npt
from ..util import np
from numbers import Number
from .indexablearray import IndexableArray
from ..raggedshape import RaggedShape, RaggedView, ViewBase, RaggedView2
from ..util import unsafe_extend_left
from ..arrayfunctions import HANDLED_FUNCTIONS, REDUCTIONS, ACCUMULATIONS


class Shape(tuple):
    def __eq__(self, other):
        return (np.all(s == o) for s, o in zip(self, other))


ShapeLike = Union[List[int], RaggedShape, RaggedView, Shape]


def reduction(allowed_axis=(None, 0, -1, 1)):
    def reduction_func(func):
        def new_func(self, axis=None, dtype=None, keepdims=False, out=None):
            assert out is None
            if axis is None:
                return getattr(np, func.__name__)(self.ravel()).item()  # force return of scalar, not object
            else:
                if axis not in allowed_axis:
                    return NotImplemented
                r = func(self, axis=axis)
                if keepdims:
                    r = r[:, None]
                return r

        return new_func

    return reduction_func


INVERSE_FUNCS = {
    np.add: (np.subtract, np.add),
    np.subtract: (np.subtract, np.add),
    np.bitwise_xor: (np.bitwise_xor, np.bitwise_xor),
}


class RaggedArray(IndexableArray, np.lib.mixins.NDArrayOperatorsMixin):
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

    def __init__(self, data: npt.ArrayLike, shape: ShapeLike = None, dtype: npt.DTypeLike = None, safe_mode: bool = True):
        if shape is None:
            data, shape = self._from_array_list(data, dtype)
        elif isinstance(shape, (ViewBase, RaggedView2)):
            shape = shape
        else:
            shape = RaggedShape.asshape(shape)
        if not hasattr(data, "__array_ufunc__"):
            data = np.asanyarray(data, dtype=dtype)
        super().__init__(data, shape)
        self._safe_mode = safe_mode

    @property
    def shape(self) -> Shape:
        return Shape((self._shape.n_rows, self._shape.lengths))

    @property
    def ndim(self) -> int:
        return 2

    @property
    def lengths(self) -> npt.ArrayLike:
        return self._shape.lengths

    def astype(self, dtype: npt.DTypeLike) -> 'RaggedArray':
        """Return ragged array with underlying data changed to dtype

        Parameters
        ----------
        dtype : npt.DTypeLike

        Returns
        -------
        'RaggedArray'

        """
        return RaggedArray(self.ravel().astype(dtype), self._shape)

    def __len__(self) -> int:
        return self._shape.n_rows

    def __iter__(self) -> types.GeneratorType:
        """Return an iterator over the rows in the ragged array

        Returns
        -------
        types.GeneratorType

        """
        flat = self.ravel()
        return (
            flat[start:start + l]
            for start, l in zip(self._shape.starts, self._shape.lengths)
        )

    def __repr__(self):
        try:
            return self._proper_repr()
        except Exception:
            return super().__repr__()

    def _proper_repr(self) -> str:
        if self.size > 100:
            rows = [str(row[:100]) for row in self[:100]]
        else:
            rows = [f"{row}" for row in self]
        text = "\n".join(rows)
        return f"ragged_array({text})"

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self[:20])

    def save(self, filename):
        """Saves the ragged array to file using np.savez

        Parameters
        ----------
        filename : str
            name of the file
        """
        np.savez(filename, data=self.ravel(), **(self._shape.to_dict()))

    @classmethod
    def load(cls, filename: str) -> 'RaggedArray':
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
        t = np.all(self.ravel() == other.ravel())
        return t and (self._shape == other._shape)

    def tolist(self) -> List[List]:
        """Return a list of list representing rows

        Returns
        -------
        List[List]

        """
        return [row.tolist() for row in self]

    def to_numpy_array(self) -> np.ndarray:
        """
        Return a normal numpy array with the same shape and data

        Requires that all the rows are of the same lengths

        Returns
        -------
        np.ndarray
        """

        if len(self) == 0:
            return np.empty(shape=(0, 0))
        L = self._shape.lengths[0]
        assert np.all(self._shape.lengths == L)
        return self.ravel().reshape(self._shape.n_rows, L)

    @classmethod
    def from_numpy_array(cls, array: npt.ArrayLike) -> 'RaggedArray':
        shape = RaggedShape.from_tuple_shape(array.shape)
        return cls(array.ravel(), shape)

    @classmethod
    def _from_array_list(cls, array_list, dtype=None):
        data = np.array(
            [element for array in array_list for element in array], dtype=dtype
        )  # This can be done faster
        return data, RaggedShape([len(a) for a in array_list])

        shape = RaggedShape([len(a) for a in array_list])
        if len(array_list) == 0:
            data = np.empty((0,), dtype=dtype)
        else:
            data = np.concatenate(array_list, dtype=dtype)
        return data, shape

    ### Broadcasting
    def _broadcast_rows(self, values, dtype=None):
        if dtype is None:
            dtype = self.dtype
        data = self._shape.broadcast_values(values, dtype=dtype)
        assert data.dtype == dtype, (values.dtype, data.dtype, dtype)
        return RaggedArray(data, self._shape)

    def _reduce(self, ufunc, ra, axis=0, **kwargs):
        assert axis in (
            1,
            -1,
        ), "Reductions on ragged arrays are only supported for the last axis"

        if self.size == 0:
            result = np.full(len(ra), fill_value=ufunc.identity)
        else:
            # if one or more of the last rows are empty,
            # ignore these when doing reduceat and pad in the end
            if self._shape.lengths[-1] == 0:
                first_last_empty_row = np.searchsorted(self._shape.starts, self._shape.starts[-1], side='left')
                result = ufunc.reduceat(self.ravel(), self._shape.starts[:first_last_empty_row])
                result = np.pad(result, (0, len(self._shape.starts)-first_last_empty_row), constant_values=ufunc.identity)
            else:
                result = ufunc.reduceat(self.ravel(), self._shape.starts)

        # hack to fix problem that reduceat does not give identity when index i == index i+1 (empty rows)
        # not necessary when ufunc does not have identity
        if ufunc.identity is not None:
            result[ra._shape.lengths == 0] = ufunc.identity

        return result

    def _reduce_invertable(self, ufunc, ra, axis, **kwargs):
        if not np.issubdtype(ra.dtype, np.integer):
            return NotImplemented
        accumulated = ufunc.accumulate(unsafe_extend_left(ra.ravel()))
        return INVERSE_FUNCS[ufunc][0](accumulated[ra._shape.ends], accumulated[ra._shape.starts])

    def _accumulate(self, ufunc, ra, axis=0, **kwargs):
        if ufunc in (np.add, np.subtract, np.bitwise_xor):
            return self._row_accumulate(ufunc)
        if ufunc not in ACCUMULATIONS:
            return NotImplemented
        return getattr(np, ACCUMULATIONS[ufunc])(ra, axis=axis, **kwargs)

    def __array_ufunc__(self, ufunc: callable, method: str, *inputs, **kwargs):
        """Handles ufuncs called on raggedarray objects.

        Supports __call__ reduce and accumulate modes. Broadcasts any
        column vectors to the shape of the raggedarray, and numbers to
        the whole array

        Parameters
        ----------
        ufunc : callable
        method : str
        *inputs :
        **kwargs :

        Raises
        ------
        TypeError

        """
        if method not in ("__call__", "reduce", "accumulate"):
            return NotImplemented
        self.ravel()
        if method == "reduce":
            return self._reduce(ufunc, inputs[0], **kwargs)
        if method == "accumulate":
            return self._accumulate(ufunc, inputs[0], **kwargs)
        datas = []
        inputs = [np.asanyarray(i) if not hasattr(i, "dtype") else i
                  for i in inputs]
        result_type = np.result_type(*(i.dtype for i in inputs))
        for input in inputs:
            if isinstance(input, Number) or (isinstance(input, np.ndarray) and input.ndim == 0):
                datas.append(input)
            elif isinstance(input, np.ndarray) or isinstance(input, list):
                broadcasted = self._broadcast_rows(input, dtype=result_type)
                datas.append(broadcasted.ravel())
            elif isinstance(input, RaggedArray):
                datas.append(input.ravel())
                if self._safe_mode and (input._shape != self._shape):
                    raise TypeError("inconsistent sizes")
            else:
                return NotImplemented
        return RaggedArray(ufunc(*datas, **kwargs), self._shape)

    def __array_function__(self, func: callable, types: List, args: List, kwargs: Dict):
        """Handles any numpy array functions called on a raggedarray

        Parameters
        ----------
        func : callable
        types : List
        args : List
        kwargs : Dict
        """
        self.ravel()
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if func != np.where and not all(issubclass(t, (self.__class__, np.ndarray)) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def fill(self, value: Number):
        ''' Fill the whole array with value'''
        self.ravel().fill(value)

    def nonzero(self) -> Tuple[npt.ArrayLike]:
        """Return the indices of all nonzero entries in the array

        Returns
        -------
        Tuple[npt.ArrayLike]
        """

        flat_indices = np.flatnonzero(self.ravel())
        return self._shape.unravel_multi_index(flat_indices)

    @reduction(allowed_axis=(None, 0, 1, -1))
    def sum(self, axis=None):
        """Calculate sum or rowsums of the array

        Parameters
        ----------
        axis : int, optional
            If `None` compute sum for whole array. If `-1` compute row sums. If `0` compute column sums.
        keepdims : bool, default=False
            If `True` return a column vector for row sums

        Returns
        -------
        int or array_like
            If `axis` is None, the sum of the whole array. If ``axis in (1, -1)``
            array containing the row sums
        """
        if axis == 0:
            _, column_indexes = self._shape.unravel_multi_index(np.arange(self.size))
            new_dtype = self.dtype
            weights = self.ravel()
            # follow numpy's rules about changing dtype to highest possible
            if np.issubdtype(self.dtype, bool):
                return np.bincount(column_indexes[self.ravel()], minlength=np.max(self.lengths))

            if np.issubdtype(self.dtype, np.integer) or np.issubdtype(self.dtype, bool):
                if np.issubdtype(self.dtype, np.signedinteger) or np.issubdtype(self.dtype, bool):
                    new_dtype = np.int64
                else:
                    new_dtype = np.uint64
                weights = weights.astype(new_dtype)

            return np.bincount(column_indexes, weights=weights, minlength=np.max(self.lengths))
            result = np.zeros(np.max(self._shape.lengths), dtype=new_dtype)
            np.add.at(result, column_indexes, self.ravel())
            return result

        return np.add.reduce(self, axis=-1)

    @reduction(allowed_axis=(1, -1))
    def prod(self, axis=-1):
        return np.multiply.reduce(self, axis=-1)

    @reduction(allowed_axis=(None, 0, 1, -1))
    def mean(self, axis=None):
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
        assert axis in (0, -1, 1)
        if not np.issubdtype(self, np.floating):
            self = self.astype(float)
        s = self.sum(axis=axis)
        if axis == 0:
            # number of elements in each column
            lengths = self.col_counts()
        else:
            lengths = self._shape.lengths

        return (s / lengths).astype(self.dtype)

    def col_counts(self):
        counts = -np.bincount(self.lengths)
        counts[0] += len(self)
        np.cumsum(counts, out=counts)
        return counts[:-1]

    @reduction(allowed_axis=(1, -1))
    def std(self, axis=-1):
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
        return NotImplemented
        self = self.astype(float)
        K = self.mean(axis=-1, keepdims=True)
        a = ((self - K) ** 2).sum(axis=-1)
        b = (self - K).sum(axis=-1) ** 2
        std = np.sqrt((a - b / self._shape.lengths) / self._shape.lengths)
        return np.where(self._shape.lengths != 1, std, 0)

    @reduction(allowed_axis=(1, -1))
    def all(self, axis=-1):
        """Check if all elements of the array are ``True``

        Returns
        -------
        bool
            Whether or not all elements evaluate to ``True``
        """
        return np.logical_and.reduce(self, axis=-1)
        nonzeros = np.flatnonzero(self.ravel().ravel())
        counts = np.searchsorted(nonzeros, self._shape.ends)-np.searchsorted(nonzeros, self._shape.starts)
        return counts == self._shape.lengths

    @reduction(allowed_axis=(1, -1))
    def any(self, axis=-1):
        """Check if any elements of the array are ``True``

        Returns
        -------
        bool
            Whether or not all elements evaluate to ``True``
        """
        return np.logical_or.reduce(self, axis=-1)

    @reduction(allowed_axis=(1, -1))
    def max(self, axis, out=None):
        assert out is None
        return np.maximum.reduce(self, axis=-1)


    @reduction(allowed_axis=(-1, 1))
    def min(self, axis=None):
        return np.minimum.reduce(self, axis=1)

    @reduction(allowed_axis=(1, -1))
    def argmax(self):
        m = self.max(axis=-1, keepdims=True)
        rows, cols = np.nonzero(self == m)
        _, idxs = np.unique(rows, return_index=True)
        return cols[idxs]

    @reduction(allowed_axis=(-1, 1))
    def argmin(self, axis=None):
        return (-self).argmax(axis=-1)

    def cumsum(self, axis: int = None, dtype: npt.DTypeLike = None) -> 'RaggedArray':
        """Return an array with cumulative sums along the given axis

        Parameters
        ----------
        axis : int
        dtype : npt.DTypeLike

        Returns
        -------
        RaggedArray
        """
        if axis is None:
            return self.ravel().cumsum(dtype=dtype)
        assert axis in (1, -1)
        if self.size == 0:
            return np.empty_like(self)
        if np.issubdtype(self.dtype, np.integer):  # in (np.int8, np.int16, np.int32, np.int64):
            cm = np.cumsum(unsafe_extend_left(self.ravel()), dtype=dtype)
            offsets = cm[self._shape.starts]
            return self.__class__(cm[1:], self._shape)-offsets[:, np.newaxis]
        raise TypeError(f"cumsum is not supported for RaggedArray of type {self.dtype}")
        cm = np.insert(self.ravel().cumsum(dtype=dtype), 0, 0)
        offsets = cm[self._shape.starts]
        # offsets = np.insert(cm[self._shape.starts[1:] - 1], 0, 0)
        ra = self.__class__(cm[1:], self._shape)
        return ra - offsets[:, None]

    def _row_accumulate(self, operator, dtype=None):
        starts = self.ravel()[self._shape.starts]
        cm = operator.accumulate(self.ravel(), dtype=dtype)
        offsets = INVERSE_FUNCS[operator][0](
            starts, cm[self._shape.starts]
        )  # TODO: This is the inverse
        ra = self.__class__(cm, self._shape)
        return INVERSE_FUNCS[operator][1](ra, offsets[:, None])

    def cumprod(self, axis=None, dtype=None):
        if axis is None:
            return self.ravel().cumprod(dtype=dtype)
        if axis in (1, -1):
            return NotImplemented

    def sort(self, axis=-1):
        if axis is None:
            return self.ravel().sort()
        if axis in (1, -1):
            args = np.lexsort((self.ravel(), self._shape.index_array()))
            return self.__class__(self.ravel()[args], self._shape)

    def _as_padded_matrix(self, fill_value=0, side='right'):
        assert side in ["left", "right"]

        ends = self._shape.ends
        starts = self._shape.starts
        max_chars = np.max(ends-starts)
        view_starts = starts if side == "right" else ends-max_chars

        indices = np.minimum(view_starts[:, None]+np.arange(max_chars), ends[-1]-1)
        array = self.ravel()[indices.ravel()]
        _starts = np.arange(starts.size)*max_chars
        _lengths = max_chars-(ends-starts)
        if side == "right":
            _starts += (ends-starts)
        zeroed, _ = RaggedView(_starts, _lengths).get_flat_indices()

        array[zeroed] = fill_value
        return array.reshape((-1, int(max_chars)))

    as_padded_matrix = _as_padded_matrix
