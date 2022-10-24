import numpy as np
from numbers import Number
from .util import unsafe_extend_left, unsafe_extend_right, unsafe_extend_left_2d
from .raggedarray import RaggedArray
from .mixin import NPSArray, NPSIndexable
import logging
logger = logging.getLogger("RunLengthArray")


class RunLengthArray(NPSIndexable, np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, events, values, do_clean=False):
        assert events[0] == 0, events
        assert len(events) == len(values)+1, (events, values, len(events), len(values))
        self._events, self._starts = (events, values)
        self._events = events.view(NPSArray)
        self._starts = events[:-1]
        self._ends = events[1:]
        self._values = values.view(NPSArray)

    def __len__(self):
        if len(self._ends) == 0:
            return 0
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

    @classmethod
    def from_intervals(cls, starts, ends, size, values=True, default_value=0):
        events = np.r_[0, np.vstack(starts, ends).T.ravel(), size]
        if isinstance(values, Number):
            values = np.tile([default_value, values], events.size//2)[:-1]
        else:
            values = np.vstack([np.broadcast(default_value, values.shape), values]).T.ravel()
            values = np.append(values, default_value)
        return cls(events, values, do_clean=True)
        
    def to_array(self):
        if len(self) == 0:
            return np.empty_like(self._values, shape=(0,))
        values = self._values
        if values.dtype == np.float64:
            values = values.view(np.uint64)
        elif values.dtype == np.float32:
            values = values.view(np.uint32)
        elif values.dtype == np.float16:
            values = values.view(np.uint16)

        array = np.zeros_like(values, shape=len(self))
        diffs = unsafe_extend_left(values)[:-1] ^ values
        diffs[0] = values[0]
        array[self._starts] = diffs
        np.bitwise_xor.accumulate(array, out=array)
        return array.view(self._values.dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method not in ("__call__"):
            return NotImplemented
        if len(inputs) == 1:
            return self.__class__(self._events, ufunc(self._values))
        assert len(inputs) == 2, f"Only unary and binary operations supported for runlengtharray {len(inputs)}"

        if isinstance(inputs[1], Number):
            return self.__class__(self._events, ufunc(self._values, inputs[1]))
        elif isinstance(inputs[0], Number):
            return self.__class__(self._events, ufunc(inputs[0], self._values))
        return self._apply_binary_func(*inputs, ufunc)

    def sum(self):
        return np.sum(np.diff(self._events)*self._values)

    def any(self):
        """TODO, this can be sped up by assuming no empty runs"""
        return np.any(np.logical_and(self._values, np.diff(self._events)))

    def all(self):
        """TODO, this can be sped up by assuming no empty runs"""
        return not np.any(np.logical_and(np.logical_not(self._values), np.diff(self._events)))

    def mean(self):
        return self.sum()/self.size()

    @staticmethod
    def remove_empty_intervals(events, values, delete_first=True):
        mask = np.flatnonzero(events[:-1] == events[1:])
        if not delete_first:
            mask += 1
        return np.delete(events, mask), np.delete(values, mask)

    @staticmethod
    def join_runs(events, values):
        mask = np.flatnonzero(values[1:] == values[:-1])+1
        return np.delete(events, mask), np.delete(values, mask)

    @classmethod
    def _apply_binary_func(cls, first, other, ufunc):
        logging.info(f"Applying ufunc {ufunc} to rla with {first._values.size} values")
        assert len(first) == len(other), (first, other)
        others_corresponding = np.searchsorted(first._events, other._events[1:-1], side="right")-1
        new_values_other = ufunc(first._values[others_corresponding], other._values[1:])
        first_corresponding = np.searchsorted(other._events, first._events[1:-1], side="right")-1
        new_values_first = ufunc(first._values[1:], other._values[first_corresponding])
        events = np.concatenate([first._events[:-1], other._events[1:]])
        values = np.concatenate([[ufunc(first._values[0], other._values[0])], new_values_first, new_values_other])
        args = np.argsort(events, kind="mergesort")
        return first.__class__(*first.join_runs(*first.remove_empty_intervals(events[args], values[args[:-1]])))

    def __apply_binary_func(self, other, ufunc):
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
        values = np.bitwise_xor.accumulate(values, axis=-1)# , out=values)
        values = values.view(result_dtype)
        assert np.all(values[0][args < self._starts.size]==self._values), (values[0][args < self._starts.size], self._values)
        assert np.all(values[1][args >= self._starts.size]==other._values), (values[1][args >= self._starts.size], other._values)
        sum_values = ufunc(values[0], values[1])
        rm_idxs = np.flatnonzero(all_events[:-1] == all_events[1:])
        all_events = np.delete(all_events, rm_idxs)
        sum_values = np.delete(sum_values, rm_idxs)
        return self.__class__(np.append(all_events, self._events[-1]), sum_values)

    def _get_position(self, idx):
        idx = np.where(idx < 0, len(self)+idx, idx)
        return self._values[np.searchsorted(self._events, idx, side="right")-1]

    def _ragged_slice(self, starts, stops):
        return RunLengthRaggedArray(*self._start_to_end(starts, stops))

    def _get_slice(self, s):
        step = 1 if s.step is None else s.step
        is_reverse = step < 0
        start = 0
        end = len(self)
        if is_reverse:
            start, end = (end-1, start-1)
        if s.start is not None:
            if s.start < 0:
                start = len(self)+s.start
            else:
                start = s.start

        if s.stop is not None:
            if s.stop < 0:
                end = len(self)+s.stop
            else:
                end = s.stop
        if is_reverse:
            start, end = (end+1, start+1)
        if start >= end:
            return np.empty_like(self._values, shape=(0,))

        subset = self.__class__(*self._start_to_end(start, end))
        assert(len(subset) == (end-start)), (subset, start, end, s)
        if step != 1:
            subset = subset._step_subset(step)

        return subset

    def __repr__(self):
        return f"RLA({repr(self._events)}, {repr(self._values)})"

    def _step_subset(self, step: int):
        """
        [0, 1, 2, 3, 4] step=3

        i=[0,1,2,3,4,5], v=[0, 1, 2, 3, 4]
        /3[0,0,0,1,1,1]
        ->[0,1,1,1,2,2] -> (i+(step-1)
        
        i=[0, 1, 2], v=[0, 3]

        [0, 0, 0], step=2
        indices=[0, 1, 2, 3], values = [0, 0, 0]
        //2 = [0, 0, 1, 1] (+1//2)= [0, 1, 1, 2], (-1//2+1) =
        [0, 1], [0, 0]
        [0, 2]
        """
        step_size = abs(step)
        indices, values = (self._events, self._values)
        if step < 0:
            indices, values = (indices[-1] - indices[::-1], values[::-1])
        indices = (indices+step_size-1)//step_size
        indices, values = self.remove_empty_intervals(indices, values)# , delete_first=False)
        indices, values = self.join_runs(indices, values)
        return self.__class__(indices, values)

    def _start_to_end(self, start, end):
        start_idx = np.searchsorted(self._events, start, side="right")-1
        end_idx = np.searchsorted(self._events, end, side="left")
        values = self._values[start_idx:end_idx]
        sub = start[:, np.newaxis] if isinstance(start, np.ndarray) else start
        events = self._events[start_idx:end_idx+1]-sub
        events[..., 0] = 0
        events[..., -1] = end-start
        return events, values

    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except ValueError:
            pass
        if isinstance(idx, tuple):
            idx = tuple(i for i in idx if i is not Ellipsis)
            assert len(idx) <= 1, idx
            if len(idx) == 0:
                return self
            if len(idx) == 1:
                idx = idx[0]
        if isinstance(idx, list):
            idx = np.asanyarray(idx)
        if isinstance(idx, Number):
            return self._get_position(idx)
        if isinstance(idx, np.ndarray):
            if idx.dtype == bool:
                idx = np.flatnonzero(idx)
            return self._get_position(idx)
        if isinstance(idx, slice):
            return self._get_slice(idx)
        if isinstance(idx, RunLengthArray):
            pass
        if idx is Ellipsis:
            return self

        # assert False, f"Invalid index for {self.__class__}: {idx}"


class RunLength2dArray:
    """Multiple run lenght arrays over the same space"""
    def __init__(self, indices: RaggedArray, values: RaggedArray, row_len: int=None):
        self._values = values
        self._indices = indices
        self._row_len = row_len

    def to_array(self):
        # assert np.all(self._indices[:, -1] == self._indices[0, -1]), self._indices
        return np.array([row.to_array() for row in self])

    def __len__(self, idx):
        return len(self._indices)

    def __getitem__(self, idx):
        events = self._indices[idx]
        if self._row_len is not None:
            events = np.append(self._indices[idx], self._row_len)
        return RunLengthArray(events, self._values[idx])

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method not in ("__call__"):
            return NotImplemented
        if len(inputs) == 1:
            return self.__class__(self._events, ufunc(self._values), self._row_len)
        assert len(inputs) == 2

        if isinstance(inputs[1], Number):
            return self.__class__(self._events, ufunc(self._values, inputs[1]), self._row_len)
        elif isinstance(inputs[0], Number):
            return self.__class__(self._events, ufunc(self._values, inputs[0]), self._row_len)
        return NotImplemented

        return self._apply_binary_func(inputs[1], ufunc)

    def any(self, axis=None):
        return self._values.any(axis=axis)

    def all(self, axis=None):
        return self._values.all(axis=axis)

    @classmethod
    def from_array(cls, array: np.ndarray):
        array = np.asanyarray(array)
        mask = unsafe_extend_left_2d(array)[:, :-1] != array
        mask[:, 0] = True
        mask[:, -1] = True
        indices = np.flatnonzero(mask)
        values = array.ravel()[indices]
        indices = RaggedArray(indices, mask.sum(axis=-1))
        values = RaggedArray(values, indices.shape)
        indices = indices-indices[:, 0][:, np.newaxis]
        return cls(indices, values, array.shape[-1])


class RunLengthRaggedArray(RunLength2dArray):
    """Multiple row-lenght arrays of differing lengths"""

    def __getitem__(self, idx):
        events = self._indices[idx]
        if self._row_len is not None:
            events = np.append(self._indices[idx], self._row_len[idx])
        return RunLengthArray(events, self._values[idx])

    def to_array(self):
        # assert np.all(self._indices[:, -1] == self._indices[0, -1]), self._indices
        return RaggedArray([row.to_array() for row in self])

    @classmethod
    def from_ragged_array(cls, ragged_array: RaggedArray):
        data = ragged_array.ravel()
        mask = unsafe_extend_left(data)[:-1] != data
        mask[ragged_array.shape.starts] = True
        indices = np.flatnonzero(mask)
        tmp = np.cumsum(unsafe_extend_left(mask))
        row_lens = tmp[ragged_array.shape.ends]-tmp[ragged_array.shape.starts]
        values = data[indices]
        indices = RaggedArray(indices, row_lens)
        start_indices = indices[:, 0][:, np.newaxis]
        indices = indices-start_indices
        return cls(indices,
                   RaggedArray(values, row_lens), ragged_array.shape.lengths)
