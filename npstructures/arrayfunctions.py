import numpy as np
HANDLED_FUNCTIONS = {}

def implements(np_function):
   "Register an __array_function__ implementation for RaggedArray objects."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator

@implements(np.concatenate)
def concatenate(ragged_arrays):
    data = np.concatenate([ra._data for ra in ragged_arrays])
    row_sizes = np.concatenate([ra.shape.lengths for ra in ragged_arrays])
    #offsets = np.cumsum(row_sizes)
    return ragged_arrays[0].__class__(data, row_sizes)

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
    return ragged_array.__class__(data, shape=shape)

@implements(np.ones_like)
def ones_like(ragged_array, dtype=None, shape=None):
    shape = ragged_array.shape if shape is None else shape
    dtype = ragged_array.dtype if dtype is None else dtype
    data = np.ones(shape.size, dtype=dtype)
    return ragged_array.__class__(data, shape=shape)
