import numpy.typing as npt
from typing import Union, List, Tuple, Dict, NewType
from numbers import Number
import numpy as np
from .raggedshape import RaggedView

RaggedArray = NewType('RaggedArray', npt.ArrayLike)


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
def concatenate(ragged_arrays: List[RaggedArray], axis: int = 0) -> RaggedArray:
    """Concatenate a set of raggedarrays along the given axis

    Parameters
    ----------
    ragged_arrays : List[RaggedArray]
    axis : int

    Returns
    -------
    RaggedArray

    """
    if axis == 0:
        data = np.concatenate([ra.ravel() for ra in ragged_arrays])
        row_sizes = np.concatenate([ra._shape.lengths for ra in ragged_arrays])
        return ragged_arrays[0].__class__(data, row_sizes)
    elif axis in [-1, 1]:
        return ragged_arrays[0].__class__([np.concatenate([row for row in rows])
                                           for rows in zip(*ragged_arrays)])
    else:
        return NotImplemented


@implements(np.diff)
def diff(ragged_array: RaggedArray, n: int = 1, axis: int = -1) -> RaggedArray:
    """Return diffs for each row in a raggedarray

    Parameters
    ----------
    ragged_array : RaggedArray
    n : int
    axis : int

    """
    if axis not in [-1, 1]:
        return NotImplemented
    d = np.diff(ragged_array.ravel(), n=n)
    lengths = np.maximum(ragged_array._shape.lengths - n, 0)
    indices, shape = RaggedView(ragged_array._shape.starts, lengths).get_flat_indices()
    return ragged_array.__class__(d[indices], shape)


@implements(np.zeros_like)
def zeros_like(ragged_array: RaggedArray, dtype: npt.DTypeLike = None, shape: Tuple[int] = None) -> RaggedArray:
    """Return a raggedarray with the same shape and dtype as raggedarray filled with zeros

    Parameters
    ----------
    ragged_array : RaggedArray
    dtype : npt.DTypeLike
    shape : Tuple[int]

    Returns
    -------
    RaggedArray

    """
    shape = ragged_array._shape if shape is None else shape
    dtype = ragged_array.dtype if dtype is None else dtype
    data = np.zeros(shape.size, dtype=dtype)
    return ragged_array.__class__(data, shape=shape)


@implements(np.ones_like)
def ones_like(ragged_array: RaggedArray, dtype: npt.DTypeLike=None, shape: Tuple[int]=None) -> RaggedArray:
    """Return a raggedarray with the same shape and dtype as raggedarray filled with ones

    Parameters
    ----------
    ragged_array : RaggedArray
    dtype : npt.DTypeLike
    shape : Tuple[int]

    Returns
    -------
    RaggedArray

    """
    shape = ragged_array._shape if shape is None else shape
    dtype = ragged_array.dtype if dtype is None else dtype
    data = np.ones(shape.size, dtype=dtype)
    return ragged_array.__class__(data, shape=shape)


@implements(np.empty_like)
def empty_like(ragged_array: RaggedArray, dtype: npt.DTypeLike=None, shape: Tuple[int]=None) -> RaggedArray:
    """Return an empty raggedarray with the same shape and dtype as raggedarray

    Parameters
    ----------
    ragged_array : RaggedArray
    dtype : npt.DTypeLike
    shape : Tuple[int]

    Returns
    -------
    RaggedArray

    """
    shape = ragged_array._shape if shape is None else shape
    dtype = ragged_array.dtype if dtype is None else dtype
    data = np.empty(shape.size, dtype=dtype)
    return ragged_array.__class__(data, shape=shape)


@implements(np.where)
def where(ragged_mask: RaggedArray, x: RaggedArray=None, y: RaggedArray=None) -> RaggedArray:
    """Perform an ifelse (tertiary operator) on a raggedarray

    Inicies where ragged_mask is True gets the corresponding value
    from x. Where False it gets from y

    Parameters
    ----------
    ragged_mask : RaggedArray
    x : RaggedArray
    y : RaggedArray

    Returns
    -------
    RaggedArray
    """
    assert (x is not None) and (y is not None), "where is only supported for ifelse for ragged_array"
    cls = x.__class__
    if not isinstance(x, Number):
        if ragged_mask.size < x.size:
            ragged_mask = x._broadcast_rows(ragged_mask)  # TODO: this is ugly, clean
        x = x.ravel()
    if not isinstance(y, Number):
        y = y.ravel()
    data = np.where(ragged_mask.ravel(), x, y)
    return cls(data, ragged_mask._shape)


@implements(np.unique)
def unique(ragged_array: RaggedArray, axis: int=None, return_counts: bool=False) -> Union[RaggedArray, Tuple[RaggedArray]]:
    """Get the unqiue values from ragged_array. If return_counts then also return the number of elemtents with each value

    Parameters
    ----------
    ragged_array : RaggedArray
    axis : int
    return_counts : bool

    Returns
    -------
    Union[RaggedArray, Tuple[RaggedArray]]
    """
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
    unique_mask[ragged_array._shape.starts] = True
    if return_counts:
        counts = np.diff(np.flatnonzero(unique_mask))
    total_counts = np.cumsum(unique_mask)
    start_counts = total_counts[ragged_array._shape.starts] - 1
    total_counts[-1] = 0  ## HAHAHACK
    end_counts = total_counts[ragged_array._shape.ends - 1]
    new_shape = end_counts - start_counts
    unique_mask = unique_mask[:-1]
    new_data = sorted_array.ravel()[unique_mask]

    ra = ragged_array.__class__(new_data, new_shape)
    if not return_counts:
        return ra
    return ra, ragged_array.__class__(counts, new_shape)



"""
   
    shape = ragged_array._shape if shape is None else shape
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
    return unique, counts.reshape(a._shape)
"""
