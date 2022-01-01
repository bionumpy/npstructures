import numpy as np
from numbers import Number
from itertools import chain
from .raggedshape import RaggedShape, CodedRaggedShape, RaggedView

HANDLED_FUNCTIONS = {}

class RaggedArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, data, shape=None, dtype=None):
        if shape is None:
            data, shape = self.from_array_list(data, dtype)
        else:
            shape = CodedRaggedShape.asanyshape(shape)
        self.shape = shape
        self._data = np.asanyarray(data)
        self.size = self._data.size
        self.dtype = self._data.dtype

    def __len__(self):
        return self.shape.n_rows

    def __iter__(self):
        return (self._data[start:end] for start, end in zip(self.shape.starts, self.shape.ends))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.tolist()})"

    def __str__(self):
        return str(self.tolist())

    def save(self, filename):
        np.savez(filename, data=self._data, **(self.shape.to_dict()))

    @classmethod
    def load(cls, filename):
        D = np.load(filename)
        shape = RaggedShape.from_dict(D)
        return cls(D["data"], shape)

    def equals(self, other):
        return self.shape == other.shape and np.all(self._data==other._data)

    def tolist(self):
        return [row.tolist() for row in self]

    @classmethod
    def from_array_list(cls, array_list, dtype=None):
        offsets = np.cumsum([0] + [len(array) for array in array_list], dtype=np.int32)
        data_size = offsets[-1]
        
        data = np.array([element for array in array_list for element in array], dtype=dtype) # This can be done faster
        return data, CodedRaggedShape(offsets)

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

    def sum(self, axis=None):
        if axis is None:
            return self._data.sum()
        if axis == -1 or axis==1:
            return np.bincount(self.shape.index_array(), self._data, minlength=self.shape.starts.size)
        return NotImplemented
 
    def mean(self, axis=None):
        s = self.sum(axis=axis)
        if axis is None:
            return s/self._data.size
        if axis == -1 or axis==1:
            return s/self.shape.lengths
        return NotImplemented

    def all(self, axis=None):
        if axis is None:
            return np.all(self._data)
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def concatenate(cls, ragged_arrays):
        data = np.concatenate([ra._data for ra in ragged_arrays])
        row_sizes = np.concatenate([[0]]+[ra.shape.lengths for ra in ragged_arrays])
        offsets = np.cumsum(row_sizes)
        return cls(data, offsets)

    def ravel(self):
        return self._data

    def nonzero(self):
        flat_indices = np.flatnonzero(self._data)
        return self.shape.unravel_multi_index(flat_indices)

def implements(np_function):
   "Register an __array_function__ implementation for RaggedArray objects."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator

@implements(np.concatenate)
def concatenate(ragged_arrays):
    data = np.concatenate([ra._data for ra in ragged_arrays])
    row_sizes = np.concatenate([[0]]+[ra.shape.lengths for ra in ragged_arrays])
    offsets = np.cumsum(row_sizes)
    return RaggedArray(data, offsets)

@implements(np.all)
def our_all(ragged_array):
    return ragged_array.all()

@implements(np.nonzero)
def nonzero(ragged_array):
    return ragged_array.nonzero()

@implements(np.zeros_like)
def zeros_like(ragged_array, dtype=None, shape=None):
    shape = ragged_array.shape if shape is None else shape
    dtype = ragged_array.dtype if dtype is None else dtype
    data = np.zeros(shape.size, dtype=dtype)
    return RaggedArray(data, shape=shape)
