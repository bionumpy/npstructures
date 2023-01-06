import cupy as cp
import numpy as np

def cp_unsafe_extend_right(array, n=1):
    assert len(array.shape) == 1
    return cp.lib.stride_tricks.as_strided(array, shape=(array.size+n, ))


def cp_unsafe_extend_left(array, n=1):
    assert len(array.shape) == 1
    return cp_unsafe_extend_right(array[::-1], n)[::-1]
