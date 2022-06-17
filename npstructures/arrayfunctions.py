import numpy as np
from .raggedshape import RaggedView


def get_ra_func(name):
    return lambda ragged_array, *args, **kwargs: getattr(ragged_array, name)(
        *args, **kwargs
    )


REDUCTIONS = {
    np.add: "sum",
    np.logical_and: "all",
    np.logical_or: "any",
    np.maximum: "max",
    np.minimum: "min",
    np.multiply: "prod",
}

ACCUMULATIONS = {np.add: "cumsum"}

NON_REDUCTION_OPERATIONS = ["nonzero", "mean", "std", "argmax", "argmin"]

HANDLED_FUNCTIONS = {
    getattr(np, name): get_ra_func(name)
    for name in list(REDUCTIONS.values())
    + list(ACCUMULATIONS.values())
    + NON_REDUCTION_OPERATIONS
}

ROW_OPERATIONS = list(REDUCTIONS.values()) + \
                 ["mean", "std", "argmax", "argmin"] + \
                 list(ACCUMULATIONS.values())




def implements(np_function):
    "Register an __array_function__ implementation for RaggedArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate(ragged_arrays, axis=0):
    if axis == 0:
        data = np.concatenate([ra._data for ra in ragged_arrays])
        row_sizes = np.concatenate([ra.shape.lengths for ra in ragged_arrays])
        return ragged_arrays[0].__class__(data, row_sizes)
    elif axis in [-1, 1]:
        return ragged_arrays[0].__class__([np.concatenate([row for row in rows])
                                           for rows in zip(*ragged_arrays)])
    else:
        return NotImplemented


@implements(np.diff)
def diff(ragged_array, n=1, axis=-1):
    if axis not in [-1, 1]:
        return NotImplemented

    # assert np.all(ragged_array.shape.lengths>=n)
    d = np.diff(ragged_array._data, n=n)
    lengths = np.maximum(ragged_array.shape.lengths - n, 0)
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


@implements(np.empty_like)
def empty_like(ragged_array, dtype=None, shape=None):
    shape = ragged_array.shape if shape is None else shape
    dtype = ragged_array.dtype if dtype is None else dtype
    data = np.empty(shape.size, dtype=dtype)
    return ragged_array.__class__(data, shape=shape)


@implements(np.unique)
def unique(ragged_array, axis=None, return_counts=False):
    if axis is None:
        return np.unique(ragged_array.ravel(), return_counts=return_counts)

    if axis not in (-1, 1):
        return NotImplemented
    if ragged_array.size == 0:
        if return_counts:
            return ragged_array, np.empty_like(ragged_array, dtype=int)
        return ragged_array
    sorted_array = ragged_array.sort()
    unique_mask = np.concatenate(
        ([True], sorted_array.ravel()[:-1] != sorted_array.ravel()[1:], [True])
    )
    unique_mask[ragged_array.shape.starts] = True
    if return_counts:
        counts = np.diff(np.flatnonzero(unique_mask))
    total_counts = np.cumsum(unique_mask)
    start_counts = total_counts[ragged_array.shape.starts] - 1
    total_counts[-1] = 0  ## HAHAHACK
    end_counts = total_counts[ragged_array.shape.ends - 1]
    new_shape = end_counts - start_counts
    unique_mask = unique_mask[:-1]
    # print(sorted_array.ravel(), unique_mask)
    new_data = sorted_array.ravel()[unique_mask]

    ra = ragged_array.__class__(new_data, new_shape)
    if not return_counts:
        return ra
    return ra, ragged_array.__class__(counts, new_shape)


"""
   
    shape = ragged_array.shape if shape is None else shape
    dtype = ragged_array.dtype if dtype is None else dtype
    data = np.empty(shape.size, dtype=dtype)
    return ragged_array.__class__(data, shape=shape)

def row_unique(a, return_counts=False):
    unique = np.sort(a)
    duplicates = unique[:,  1:] == unique[:, :-1]
    unique[:, 1:][duplicates] = 0
    if not return_counts:
        return unique
    count_matrix = np.zeros(a.size, dtype="int")
    idxs = np.flatnonzero(unique)
    counts = np.diff(idxs)
    count_matrix[idxs[:-1]] = counts
    count_matrix[idxs[-1]] = a.size-idxs[-1]
    return unique, counts.reshape(a.shape)
"""
