import numpy as np
from .raggedshape import RaggedView

def get_ra_func(name):
   return lambda ragged_array, *args, **kwargs: getattr(ragged_array, name)(*args, **kwargs)
REDUCTIONS = {np.add: "sum",
              np.logical_and: "all",
              np.logical_or: "any",
              np.maximum: "max",
              np.minimum: "min",
              np.multiply: "prod"}

ACCUMULATIONS = {np.multiply: "cumprod",
                 np.add: "cumsum"}

HANDLED_FUNCTIONS = {getattr(np, name): get_ra_func(name) for name in 
                     list(REDUCTIONS.values()) + list(ACCUMULATIONS.values()) + ["nonzero", "mean", "std", "argmax", "argmin"]}

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
    return ragged_arrays[0].__class__(data, row_sizes)

@implements(np.diff)
def diff(ragged_array, n=1, axis=-1):
   assert axis in (-1, 1)
   # assert np.all(ragged_array.shape.lengths>=n)
   d = np.diff(ragged_array._data, n=n)
   lengths = np.maximum(ragged_array.shape.lengths-n, 0)
   indices, shape = RaggedView(ragged_array.shape.starts, lengths).get_flat_indices()
   return ragged_array.__class__(d[indices], shape)
   

# @implements(np.all):
# def our_all(ragged_array, *args, **kwargs):
#     return ragged_array.all(*args, **kwargs)
# 
# @implements(np.sum):
# def our_sum(ragged_array, *args, **kwargs):
#     return ragged_array.sum(*args, **kwargs)
# 
# @implements(np.nonzero)
# def nonzero(ragged_array):
#     return ragged_array.nonzero()

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
