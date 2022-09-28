from . import RaggedArray, RaggedView
import numpy as np


def ragged_slice(array, starts=None, ends=None):
    assert isinstance(array, RaggedArray)
    if starts is None:
        starts = array.shape.starts
    else:
        starts = array.shape.starts+starts
    if ends is None:
        ends = array.shape.ends
    else:
        ends = np.where(ends < 0, array.shape.ends+ends, array.shape.starts+ends)
    lengths = ends-starts
    indices, shape = RaggedView(starts, lengths).get_flat_indices()
    return array.__class__(array.ravel()[indices], shape)


class RaggedSlice:
    def __init__(self, starts=None, ends=None, steps=None):
        pass
