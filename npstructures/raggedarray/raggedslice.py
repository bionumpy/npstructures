from . import RaggedArray, RaggedView
import numpy as np


def ragged_slice(array, starts=None, ends=None):
    if isinstance(array, RaggedArray):
        base_starts = array.shape.starts
        base_ends = array.shape.ends
    else:
        assert isinstance(array, np.ndarray)
        if len(array.shape) == 1:
            base_starts = 0
            base_ends = array.size
        else:
            assert len(array.shape) == 2
            base_starts = np.arange(len(array))*array.shape[-1]
            base_ends = base_starts+array.shape[-1]
    if starts is None:
        starts = base_starts
    else:
        starts = base_starts+starts
    if ends is None:
        ends = base_ends
    else:
        ends = np.where(ends < 0, base_ends+ends, base_starts+ends)
    lengths = ends-starts
    indices, shape = RaggedView(starts, lengths).get_flat_indices()
    return array.__class__(array.ravel()[indices], shape)


class RaggedSlice:
    def __init__(self, starts=None, ends=None, steps=None):
        pass
