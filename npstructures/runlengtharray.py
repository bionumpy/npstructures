import numpy as np
from .util import unsafe_extend_left, unsafe_extend_right


class RunLengthArray:
    def __init__(self, starts, ends, values):
        self._starts = starts
        self._ends = ends
        self._values = values

    @classmethod
    def from_array(cls, array):
        array = np.asanyarray(array)
        mask = unsafe_extend_left(array) != unsafe_extend_right(array)
        mask[0], mask[-1] = (True, True)
        indices = np.flatnonzero(mask)
        starts, ends = (indices[:-1], indices[1:])
        values = array[starts]
        return cls(starts, ends, values)

    def __len__(self):
        return self._ends[-1]

    def to_array(self):
        array = np.zeros_like(self._values, shape=len(self))
        diffs = np.diff(unsafe_extend_left(self._values))
        diffs[0] = self._values[0]
        array[self._starts] = diffs
        np.cumsum(array, out=array)
        return array
