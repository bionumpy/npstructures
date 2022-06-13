import numpy as np


def unsafe_extend_right(array, n=1):
    assert len(array.shape) == 1
    return np.lib.stride_tricks.as_strided(array, shape=(array.size+n, ), writeable=False)


def unsafe_extend_left(array, n=1):
    assert len(array.shape) == 1
    return unsafe_extend_right(array[::-1], n)[::-1]
