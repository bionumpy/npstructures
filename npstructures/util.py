import numpy as np


def as_strided(array, *args, **kwargs):
    if hasattr(array, "as_strided"):
        return array.as_strided(*args, **kwargs)
    assert not np.issubdtype(array.dtype, np.object_)
    return np.lib.stride_tricks.as_strided(array, *args, **kwargs)


def unsafe_extend_right(array, n=1):
    assert len(array.shape) == 1
    return as_strided(array, shape=(array.size+n, ), writeable=False, subok=True)


def unsafe_extend_left(array, n=1):
    assert len(array.shape) == 1
    return unsafe_extend_right(array[::-1], n)[::-1]
