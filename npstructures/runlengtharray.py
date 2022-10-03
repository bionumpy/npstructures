import numpy as np
from numbers import Number
from .util import unsafe_extend_left, unsafe_extend_right


class RunLengthArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, events, values):
        assert events[0] == 0, events
        assert len(events) == len(values)+1, (events, values, len(events), len(values))
        self._events = events
        self._starts = events[:-1]
        self._ends = events[1:]
        self._values = values

    def __len__(self):
        return self._ends[-1]

    @property
    def starts(self):
        return self._events[:-1]

    @property
    def ends(self):
        return self._events[1:]

    @property
    def values(self):
        return self._values

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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method not in ("__call__"):
            return NotImplemented
        assert len(inputs) == 2
        if isinstance(inputs[1], Number):
            return self.__class__(self._events, ufunc(self._values, inputs[1]))
        return self._apply_binary_func(inputs[1], ufunc)

    def _apply_binary_func(self, other, ufunc):
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
        sum_values = ufunc(values[0], values[1])
        rm_idxs = np.flatnonzero(all_events[:-1] == all_events[1:])
        all_events = np.delete(all_events, rm_idxs)
        sum_values = np.delete(sum_values, rm_idxs)
        return self.__class__(np.append(all_events, self._events[-1]), sum_values)

    def _get_position(self, idx):
        return self._values[np.searchsorted(self._events, idx, side="right")-1]

    def _get_slice(self, s):
        assert s.step is None
        start = 0
        end = len(self)
        if s.start is not None:
            if s.start < 0:
                start = len(self)+start
            else:
                start = s.start

        if s.stop is not None:
            if s.stop < 0:
                end = len(self)+end
            else:
                end = s.stop
        return self._start_to_end(start, end)

    def _start_to_end(self, start, end):
        start_idx = np.searchsorted(self._events, start, side="right")-1
        end_idx = np.searchsorted(self._events, end, side="left")
        values = self._values[start_idx:end_idx]
        events = self._events[start_idx:end_idx+1]-start
        events[0] = 0
        events[-1] = end-start
        return self.__class__(events, values)

    def __getitem__(self, idx):
        if isinstance(idx, Number):
            return self._get_position(idx)
        if isinstance(idx, slice):
            return self._get_slice(idx)
