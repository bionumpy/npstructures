import numpy as np
from .util import unsafe_extend_left, unsafe_extend_right


class RunLengthArray:
    def __init__(self, events, values):
        assert events[0] == 0, events
        assert len(events)==len(values)+1, (events, values)
        self._events = events
        self._starts = events[:-1]
        self._ends = events[1:]
        self._values = values

    def __len__(self):
        return self._ends[-1]

    @classmethod
    def from_array(cls, array):
        array = np.asanyarray(array)
        mask = unsafe_extend_left(array) != unsafe_extend_right(array)
        mask[0], mask[-1] = (True, True)
        indices = np.flatnonzero(mask)
        values = array[indices[:-1]]
        return cls(indices, values)

    def to_array(self):
        array = np.zeros_like(self._values, shape=len(self))
        diffs = np.diff(unsafe_extend_left(self._values))
        diffs[0] = self._values[0]
        array[self._starts] = diffs
        np.cumsum(array, out=array)
        return array

    def __add__(self, other):
        assert len(self) == len(other), (self, other)
        all_events = np.concatenate([self._starts, other._starts])
        args = np.argsort(all_events, kind="mergesort")
        all_events = all_events[args]

        result_dtype = np.result_type(*(i.dtype for i in (self._values, other._values)))
        temp_dtype = result_dtype
        if result_dtype == np.float64:
            temp_dtype = np.uint64
        elif result_dtype == np.float32:
            temp_dtype = np.uint32
        o_values = other._values.astype(result_dtype).view(temp_dtype)
        m_values = self._values.astype(result_dtype).view(temp_dtype)
        other_d = unsafe_extend_left(o_values[:-1]) ^ o_values
        other_d[0] = o_values[0]
        values = np.zeros((2, len(all_events)), dtype=temp_dtype)

        my_d = unsafe_extend_left(m_values)[:-1] ^ m_values
        my_d[0] = m_values[0]
        values[1, args >= self._starts.size] = other_d
        values[0, args < self._starts.size] = my_d
        np.bitwise_xor.accumulate(values, axis=-1, out=values)
        values = values.view(result_dtype)
        sum_values = values[0]+values[1]
        rm_idxs = np.flatnonzero(all_events[:-1] == all_events[1:])
        all_events = np.delete(all_events, rm_idxs)
        sum_values = np.delete(sum_values, rm_idxs)
        return self.__class__(np.append(all_events, self._events[-1]), sum_values)
