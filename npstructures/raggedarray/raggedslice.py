from . import RaggedArray, RaggedView
from ..util import np


def ragged_slice(array, starts=None, ends=None):
    if isinstance(array, RaggedArray):
        array.ravel() #This is a MESS
        base_starts = array._shape.starts
        base_ends = array._shape.ends
    else:
        assert hasattr(array, "shape") and hasattr(array, "size")
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
        ends = np.where(ends < 0,
                        base_ends+ends,
                        np.minimum(base_starts+ends, base_ends))
    lengths = np.maximum(ends-starts, 0)
    indices, shape = RaggedView(starts, lengths).get_flat_indices()
    cls = RaggedArray if not isinstance(array, RaggedArray) else array.__class__
    return cls(array.ravel()[indices], shape)
