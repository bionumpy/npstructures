import numpy as np


def as_strided(array, *args, **kwargs):
    if hasattr(array, "as_strided"):
        return array.as_strided(*args, **kwargs)
    assert not np.issubdtype(array.dtype, np.object_)
    return np.lib.stride_tricks.as_strided(array, *args, **kwargs)


def unsafe_extend_right(array, n=1):
    assert len(array.shape) == 1, array.shape
    return as_strided(array, shape=(array.size+n, ), writeable=False, subok=True)


def unsafe_extend_right_2d(array, n=1):
    assert len(array.shape) == 2, array.shape
    return as_strided(array, shape=(array.shape[0], array.shape[1]+n), writeable=False, subok=True)


def unsafe_extend_left(array, n=1):
    assert len(array.shape) == 1, array.shape
    return unsafe_extend_right(array[::-1], n)[::-1]


def unsafe_extend_left_2d(array, n=1):
    assert len(array.shape) == 2
    return unsafe_extend_right_2d(array[:, ::-1], n)[:, ::-1]


def bincount2d(array):
    max_val = np.max(array)+1
    n_rows, n_cols = array.shape
    counts = np.zeros((n_rows, max_val), dtype=int)
    np.add.at(counts, (np.broadcast_to(np.arange(n_rows)[:, np.newaxis], array.shape), array), 1)
    return counts
