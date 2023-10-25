import numpy as np
from typing import Dict, List, Tuple, Union
from numpy.typing import ArrayLike, DTypeLike
from numbers import Number
from .util import unsafe_extend_left, unsafe_extend_right, unsafe_extend_left_2d
from .raggedarray import RaggedArray
from .mixin import NPSArray, NPSIndexable
from .raggedarray.raggedslice import ragged_slice
import logging
logger = logging.getLogger("RunLengthArray")


def get_ra_func(name):
    return lambda ragged_array, *args, **kwargs: getattr(ragged_array, name)(
        *args, **kwargs
    )


HANDLED_FUNCTIONS = {np.all: get_ra_func('all'),
                     np.sum: get_ra_func('sum'),
                     np.any: get_ra_func('any'),
                     np.mean: get_ra_func('mean')}


def implements(np_function):
    "Register an __array_function__ implementation for RunLengthArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.histogram)
def histogram(rla, bins=10, range=None, density=None, weights=None):
    lens = np.diff(rla._events)
    if weights is None:
        weights = lens
    else:
        weights = lens*weights
    return np.histogram(rla._values, bins, range, density, weights)


@implements(np.concatenate)
def concatenate(rl_arrays):
    sizes = [array.size for array in rl_arrays]
    offsets = np.insert(np.cumsum(sizes), 0, 0)
    events = np.concatenate(
        [offset + array.starts for array, offset in zip(rl_arrays, offsets)] + [offsets[-1:]])
    values = np.concatenate([array.values for array in rl_arrays])
    return rl_arrays[0].__class__(events, values)


class IndexableMixin:
    def _step_subset(self, step, indices, values):
        if step < 0:
            indices = indices[..., -1][..., np.newaxis]-indices[..., ::-1]
            values = values[..., ::-1]
        step_size = abs(step)
        if step_size != 1:
            indices = (indices+step_size-1)//step_size
        indices, values = self.remove_empty_intervals(indices, values)
        return indices, values

    def _getitem_tuple(self, raw_idx):
        if len(raw_idx) == 1:
            return self[raw_idx[0]]
        idx = tuple(r for r in raw_idx if r is not Ellipsis)
        assert len(idx) < 3, idx
        if len(idx) == 0:
            return self
        if len(idx) == 1 and raw_idx[0] is Ellipsis:
            idx = (slice(None),) + idx
        elif len(idx) == 1 and raw_idx[1] is Ellipsis:
            idx = idx + (slice(None),)

        if len(idx) == 2:
            row_idx, col_idx = idx
            rows = self[row_idx]
            if isinstance(row_idx, Number):
                return rows[col_idx]
            rows._indices.ravel()
            rows._values.ravel()
            if len(rows) == 0:
                return rows
            if isinstance(col_idx, Number):
                if col_idx < 0:
                    col_idx = (rows._indices[:, -1]+col_idx)[:, np.newaxis]
                mask = (rows._indices[:, :-1] <= col_idx) & (rows._indices[:, 1:] > col_idx)
                return rows._values[mask]
            elif isinstance(col_idx, slice):
                start, stop, step = (col_idx.start, col_idx.stop, col_idx.step)
                if step is None:
                    step = 1
                s = slice(start, stop, np.sign(step))
                is_reverse = step < 0
                start_col, stop_col = (None, None)
                if (start is not None and stop is not None):
                    if (is_reverse and (start <= stop) and start > 0) or (not is_reverse and (start >= stop) and stop>0):
                        return self.__class__(np.zeros((len(rows), 1), dtype=int), np.empty((len(rows), 0), dtype=int))

                if stop is not None:
                    if stop >= 0:
                        stop = np.minimum(rows.shape[-1], stop)[:, np.newaxis]
                    else:
                        stop = np.maximum(0, rows.shape[-1]+stop)[:, np.newaxis]
                    if is_reverse:
                        mask = (rows._indices[..., :-1] <= stop+1) & (rows._indices[..., 1:] > stop+1)
                        start_col = rows._indices.shape[-1]-1# np.full(len(rows), -1,  dtype=int)# np.zeros(len(rows), dtype=int)
                        r, c = np.nonzero(mask)
                        start_col[r] = c
                    else:
                        mask = (rows._indices[..., 1:] >= stop) & (rows._indices[..., :-1] < stop)
                        stop_col = np.full(len(rows), -1,  dtype=int)
                        r, c = np.nonzero(mask)
                        stop_col[r] = c
                if start is not None:
                    if start >= 0:
                        start = np.minimum(rows.shape[-1], start)[:, np.newaxis]
                    else:
                        start = np.maximum(0, rows.shape[-1]+start)[:, np.newaxis]
                    if is_reverse:
                        mask = (rows._indices[..., 1:] >= start+1) & (rows._indices[..., :-1] < start+1)
                        _, stop_col = np.nonzero(mask)
                    else:
                        mask = (rows._indices[..., :-1] <= start) & (rows._indices[..., 1:] > start)
                        _, start_col = np.nonzero(mask)
                is_empty = None
                if start_col is None and stop_col is None:
                    i, v = rows._indices, rows._values
                else:
                    s = start_col # if start_col is not None else None
                    e = stop_col+2 if stop_col is not None else None
                    e2 = stop_col+1 if stop_col is not None else None

                    if s is not None and e is not None:
                        is_empty = (s >= e) #changed from ==
                        e = np.maximum(s+1, e)
                        e2 = np.maximum(s, e2)
                    i = ragged_slice(rows._indices, starts=s,
                                     ends=e)

                    v = ragged_slice(rows._values, starts=s, # start_col if start_col is not None else None,
                                     ends=e2)
                    if is_reverse:
                        if start is not None:
                            i[:, -1] = (start+1).ravel()
                        if stop is not None:
                            i[:, 0] = (stop+1).ravel()
                    else:
                        if stop is not None:
                            i[:, -1] = stop.ravel()
                        if start is not None:
                            i[:, 0] = start.ravel()
                    i = i - i[:, 0][:, np.newaxis]
                i, v = rows._step_subset(step, i, v)
                if is_empty is not None:
                    i[is_empty, 0] = 0
                return self.__class__(i, v)

            if isinstance(idx[0], np.ndarray):
                row_idx, col_idx = (np.asanyarray(x).ravel() for x in idx)
            tmp_array = self.__class__(self._indices[row_idx], self._values[row_idx], row_len=self._row_len)
            values = [rla[c] for rla, c in zip(tmp_array, col_idx.ravel())]
            if isinstance(idx[0], np.ndarray):
                values = np.reshape(values, idx[0].shape)
            return values
        assert not isinstance(idx, tuple), (raw_idx, idx)
        return self[idx]

    def __getitem__(self, raw_idx: Union[Tuple, int, List[int], ArrayLike]):
        if isinstance(raw_idx, tuple):
            return self._getitem_tuple(raw_idx)
        idx = raw_idx
        self._values.ravel()
        self._indices.ravel()
        events = self._indices[idx]
        values = self._values[idx]
        if isinstance(events, RaggedArray):
            assert self._row_len is None or isinstance(self._row_len, Number), self._row_len
            return self.__class__(events, values, self._row_len)
        if self._row_len is not None:
            assert self._row_len is None or isinstance(self._row_len, Number), self._row_len
            events = np.append(self._indices[idx], self._row_len)
        return RunLengthArray(events, values)


class RunLengthArray(NPSIndexable, np.lib.mixins.NDArrayOperatorsMixin):
    """Class for Run Length arrays

    This is used to represent data where changes in values are
    rare. Behaves much like a normal NumPy array.

    Should be constructed using one of the classmethods, not initialized directly
    """
    def __init__(self, events, values, do_clean=False):
        self._events, self._starts = (events, values)
        self._events = events# .view(NPSArray)
        self._starts = events[:-1]
        self._ends = events[1:]
        self._values = values# .view(NPSArray)
        assert events[0] == 0, events
        assert len(events) == len(values)+1, (events, values, len(events), len(values))
        assert np.all(events[1:] > events[:-1]), f"Empty run not allowed in RunLenghtArray (use remove_empty_intervals): {events}"

    def __array__(self, dtype=None):
        return np.asanyarray(self.to_array(), dtype=dtype)

    def __len__(self) -> int:
        if len(self._ends) == 0:
            return 0
        return self._ends[-1]

    def __str__(self) -> str:
        if self.size <= 1000:
            return str(self.to_array())
        return f"[ {' ' .join(str(c) for c in self[:3].to_array())} ... {' '.join(str(c) for c in self[-3:].to_array())}]"

    def __repr__(self) -> str:
        if self.size <= 1000:
            return repr(self.to_array())

    @property
    def size(self) -> int:
        """
        Size of the array
        """
        return self._ends[-1]

    @property
    def shape(self) -> Tuple[int]:
        return (self.size,)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def dtype(self) -> DTypeLike:
        return self._values.dtype

    @property
    def starts(self) -> ArrayLike:
        """
        All idxs where the value changes
        """
        return self._events[:-1]

    @property
    def ends(self) -> ArrayLike:
        """All indices that ends a run of equal values
        """
        return self._events[1:]

    @property
    def values(self) -> ArrayLike:
        """All values that are not equal to previous value

        Returns
        -------
        ArrayLike
        """
        return self._values

    @classmethod
    def from_array(cls, array: ArrayLike) -> 'RunLengthArray':
        """Construct a RunLengthArray from a normal numpy array

        Parameters
        ----------
        array : ArrayLike
            Normal numpy array

        Returns
        -------
        'RunLengthArray'

        """
        array = np.asanyarray(array)
        mask = unsafe_extend_left(array) != unsafe_extend_right(array)
        mask[0], mask[-1] = (True, True)
        indices = np.flatnonzero(mask)
        values = array[indices[:-1]]
        return cls(indices, values)

    @classmethod
    def __from_intervals(cls, starts: ArrayLike, ends: ArrayLike, size: int , values: ArrayLike = True, default_value=0) -> 'RunLengthArray':
        """Constuct a runlength array from a set of intervals and values

        Parameters
        ----------
        starts : ArrayLike
        ends : ArrayLike
        size : int
        values : ArrayLike
        default_value :

        Returns
        -------
        'RunLengthArray'
        """

        assert np.all(ends > starts)
        assert np.all(starts[1:] > ends[:-1])
        prefix = [0] if (len(starts) == 0 or starts[0] != 0) else []
        postfix = [size] if (len(ends) == 0 or ends[-1] != size) else []
        events = np.concatenate([np.array(prefix, dtype=int), np.vstack((starts, ends)).T.ravel(), np.array(postfix, dtype=int)])
        if isinstance(values, Number):
            values = np.tile([default_value, values], events.size//2+1)
        else:
            values = np.vstack([np.broadcast(default_value, values.shape), values]).T.ravel()
            if ends[-1] != size:
                values = np.append(values, default_value)
        if (len(starts) > 0) and starts[0] == 0:
            values = values[1:]
        values = values[:(len(events)-1)]
        return cls(events, values, do_clean=True)

    def to_array(self) -> np.ndarray:
        """Convert the runlength array to a normal numpy array

        Returns
        -------
        np.ndarray
        """
        if len(self) == 0:
            return np.empty_like(self._values, shape=(0,))
        values = np.asarray(self._values)
        if values.dtype == np.float64:
            values = values.view(np.uint64)
        elif values.dtype == np.float32:
            values = values.view(np.uint32)
        elif values.dtype == np.float16:
            values = values.view(np.uint16)

        array = np.zeros_like(values, shape=len(self))
        op = np.logical_xor if array.dtype == bool else np.bitwise_xor
        diffs = op(values[:-1], values[1:])
        assert np.all(self._starts[1:] > self._starts[:-1])
        array[self._starts[1:]] = diffs
        array[self._starts[0]] = values[0]
        tmp = array.copy()
        op.accumulate(array, out=tmp)
        # array = op.accumulate(array, dtype=array.dtype)
        return tmp.view(self._values.dtype)

    def __array_function__(self, func: callable, types: List, args: List, kwargs: Dict):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, RunLengthArray) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc: callable, method: str, *inputs, **kwargs):
        """Handle numpy unfuncs called on the runlength array

        Currently only handles '__call__' modes and unary and binary functions

        Parameters
        ----------
        ufunc : callable
        method : str
        *inputs :
        **kwargs :
        """
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

    def sum(self, axis=-1, out=None):
        return np.sum(np.diff(self._events)*self._values)

    def any(self, axis=-1, out=None):
        """TODO, this can be sped up by assuming no empty runs"""
        return np.any(self._values)

    def all(self, axis=-1, out=None):
        """TODO, this can be sped up by assuming no empty runs"""
        return np.all(self._values)

    def max(self, axis=-1, out=None):
        return self._values.max()

    def astype(self, dtype: DTypeLike) -> 'RunLengthArray':
        return self.__class__(self._events, self._values.astype(dtype))

    def mean(self, axis=-1, dtype=None, out=None):
        assert dtype is None and out is None
        if np.issubdtype(self.dtype, np.integer):
            return self.astype(float).mean(axis, dtype, out)
        return self.sum()/self.size

    @staticmethod
    def remove_empty_intervals(events: ArrayLike, values: ArrayLike, delete_first: bool = True) -> Tuple[ArrayLike, ArrayLike]:
        """Remove any empty runs from a pair of indices and values.

        Should be run before creating a runlength array from the
        events and values

        Parameters
        ----------
        events : ArrayLike
        values : ArrayLike
        delete_first : bool

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
        """
        mask = np.flatnonzero(events[:-1] == events[1:])
        if not delete_first:
            mask += 1
        return np.delete(events, mask), np.delete(values, mask)

    @staticmethod
    def join_runs(events: ArrayLike, values: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Join succesive runs with the same value

        Parameters
        ----------
        events : ArrayLike
        values : ArrayLike

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
        """
        mask = np.flatnonzero(values[1:] == values[:-1])+1
        return np.delete(events, mask), np.delete(values, mask)

    @classmethod
    def _apply_binary_func(cls, first, other, ufunc):
        logging.debug(f"Applying ufunc {ufunc} to rla with {first._values.size} values")
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

    def _get_slice(self, s: slice) -> 'RunLengthArray':
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
            return self.__class__(np.array([0]), np.empty_like(self._values, shape=(0,)))

        subset = self.__class__(*self._start_to_end(start, end))
        assert(len(subset) == (end-start)), (subset, start, end, s)
        if step != 1:
            subset = subset._step_subset(step)

        return subset

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
        if isinstance(start_idx, np.ndarray):
            values = ragged_slice(self._values, start_idx, end_idx)
        else:
            values = self._values[start_idx:end_idx]
        sub = start[:, np.newaxis] if isinstance(start, np.ndarray) else start
        if isinstance(start_idx, np.ndarray):
            events = ragged_slice(self._events, start_idx, end_idx+1)-sub
        else:
            events = self._events[start_idx:end_idx+1]-sub
        events[..., 0] = 0
        events[..., -1] = end-start
        return events, values

    def _getitem_bool(self, idx: 'RunLengthArray'):
        starts = idx.starts[idx.values]
        ends = idx.ends[idx.values]
        return RunLengthRaggedArray(*self._start_to_end(starts, ends)).ravel()

    def __getitem__(self, idx: Union[Tuple, List[int], Number, ArrayLike, slice]):
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
            assert idx.dtype == bool
            return self._getitem_bool(idx)
        if idx is Ellipsis:
            return self
        assert False, f"Invalid index for {self.__class__}: {idx}"


class RunLength2dArray(IndexableMixin, np.lib.mixins.NDArrayOperatorsMixin):
    ''' Multiple RunLengthArrays of the same size. Behaves like a 2d numpy array'''
    def __init__(self, indices: RaggedArray, values: RaggedArray, row_len: int=None):
        self._values = values
        self._indices = indices
        self._row_len = row_len

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self[:min(20, len(self))])

    @property
    def shape(self) -> Tuple[int]:
        return (len(self._indices), self._row_len)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return self.shape[0]*self.shape[1]

    @property
    def dtype(self) -> DTypeLike:
        return self._values.dtype()

    def to_array(self) -> ArrayLike:
        ''' Convert to a normal 2d numpy array '''
        return np.array([row.to_array() for row in self])

    def __len__(self) -> int:
        return len(self._indices)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy unfuncs called on the runlength array

        Currently only handles '__call__' modes and unary and binary functions

        Parameters
        ----------
        ufunc : callable
        method : str
        *inputs :
        **kwargs :
        """
        if method not in ("__call__"):
            return NotImplemented
        if len(inputs) == 1:
            return self.__class__(self._indices, ufunc(self._values), self._row_len)
        assert len(inputs) == 2

        if isinstance(inputs[1], (Number, np.ndarray)):
            return self.__class__(self._indices, ufunc(self._values, inputs[1]), self._row_len)
        elif isinstance(inputs[0], (Number, np.ndarray)):
            return self.__class__(self._indices, ufunc(self._values, inputs[0]), self._row_len)
        return NotImplemented

    def any(self, axis: int = None, out=None) -> ArrayLike:
        if axis in (0, -2):
            return self._col_any()
        return self._values.any(axis=axis)

    def all(self, axis: int= None, out=None) -> ArrayLike:
        return self._values.all(axis=axis)

    def sum(self, axis: int = None, out=None) -> ArrayLike:
        if axis in (0, -2):
            return self._col_sum()
        assert (axis == -1 or axis is None)
        lens = (self._indices[:, 1:]-self._indices[:, :-1])
        if self._row_len is None:
            return np.sum(self._values*lens, axis=-1)
        internal_sum = np.sum(self._values[:, :-1] * lens, axis=-1)
        return internal_sum + self._values[:, -1]*(self._row_len-self._indices[:, -1])

    def _col_sum(self):
        positions = self._indices.ravel()
        L = np.max(positions) if self._row_len is None else self._row_len
        args = np.argsort(positions, kind="mergesort")
        if self._row_len is None:
            c = np.concatenate([self._values, np.zeros((len(self), 1),
                                                       dtype=self._values.dtype)], axis=-1)
            values = c.ravel()
        else:
            values = self._values.ravel()
        assert len(values) == len(positions), (values, positions)
        if np.issubdtype(values.dtype, np.integer):
            if np.issubdtype(values.dtype, np.signedinteger):
                values = values.astype(int)
            else:
                values = values.astype(np.uint64)

        values = np.diff(unsafe_extend_left(values))
        values[np.insert(np.cumsum(self._indices.lengths), 0, 0)[:-1]] = self._values[:, 0]
        values = values[args]
        np.cumsum(values, out=values)
        positions = positions[args]
        positions = np.append(positions, L)
        return RunLengthArray(*RunLengthArray.remove_empty_intervals(positions, values))

    def _col_any(self):
        values = self._values.astype(bool)
        indices, values = self.join_runs(self._indices, values)
        assert not np.any(values[:, :-1] == values[:, 1:]), values
        starts = indices.ravel()[values.ravel()]
        starts.sort(kind="mergesort")
        ends = indices[:, 1:].ravel()[~values[:, 1:].ravel()]
        ends.sort(kind="mergesort")
        ends = np.maximum.accumulate(ends)
        ends = np.pad(ends, (0, starts.size-ends.size), constant_values=self._row_len)
        valid_mask = unsafe_extend_right(starts) > unsafe_extend_left(ends)
        valid_mask[[0, -1]] = True
        starts = starts[valid_mask[:-1]]
        ends = ends[valid_mask[1:]]
        indices = np.array([starts, ends]).T.ravel()
        values = np.tile([True, False], len(starts))
        if len(starts) == 0 or starts[0] != 0:
            indices = np.insert(indices, 0, 0)
            values = np.insert(values, 0, False)
        if indices[-1] == self._row_len:
            values = values[:-1]
        else:
            indices = np.append(indices, self._row_len)
        return RunLengthArray(indices, values)

    @classmethod
    def from_array(cls, array: ArrayLike) -> 'RunLength2dArray':
        """Construct a Runlength2dArray from a normal 2d numpy array

        Parameters
        ----------
        array : ArrayLike

        Returns
        -------
        'RunLength2dArray'


        """
        array = np.asanyarray(array)
        mask = unsafe_extend_left_2d(array)[:, :-1] != array
        mask[:, 0] = True
        mask[:, -1] = True
        indices = np.flatnonzero(mask)
        values = array.ravel()[indices]
        indices = RaggedArray(indices, mask.sum(axis=-1))
        values = RaggedArray(values, indices._shape)
        indices = indices-indices[:, 0][:, np.newaxis]
        return cls(indices, values, array.shape[-1])

    @staticmethod
    def join_runs(indices: ArrayLike, values: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        mask = unsafe_extend_left(values.ravel())[:-1] != values.ravel()
        mask[values._shape.starts] = True
        shape = RaggedArray(mask, values._shape).sum(axis=-1)
        indices = RaggedArray(indices.ravel()[mask], shape)
        values = RaggedArray(values.ravel()[mask], shape)
        return indices, values

    @classmethod
    def from_intervals(cls, starts: ArrayLike, ends: ArrayLike, row_len: int, value=1) -> 'RunLength2dArray':
        """Construct a RunLength2dArray from a set of intervals

        Each interval becomes a row in the 2d array

        Parameters
        ----------
        cls :
        starts : ArrayLike
        ends : ArrayLike
        row_len : int
        value :

        Returns
        -------
        'RunLength2dArray'

        """
        starts_after_zero = starts > 0
        ends_before_end = ends < row_len
        value = np.asanyarray(value)
        shape = starts_after_zero + 1 + ends_before_end
        indices = RaggedArray(np.zeros(shape.sum(), int), shape)
        values = RaggedArray(np.zeros(shape.sum(), value.dtype), shape)
        indices[np.arange(len(starts)), starts_after_zero.astype(int)] = starts
        values[np.arange(len(starts)), starts_after_zero.astype(int)] = value
        indices[ends_before_end, -1] = ends[ends_before_end]
        return cls(indices, values, row_len)


class RunLengthRaggedArray(RunLength2dArray, IndexableMixin):
    """Multiple row-lenght arrays of differing lengths"""

    @property
    def shape(self) -> Tuple[int]:
        if self._row_len is None:
            return (len(self._indices), self._indices[..., -1])
        return (len(self._indices), self._row_len)


    @staticmethod
    def remove_empty_intervals(events: ArrayLike, values: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Remove any empty runs from a pair of indices and values.

        Should be run before creating a runlength array from the
        events and values

        Parameters
        ----------
        events : ArrayLike
        values : ArrayLike
        delete_first : bool

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
        """
        values.ravel()
        mask = events[..., :-1] != events[..., 1:]
        mask2 = np.ones_like(events, dtype=bool)
        mask2[..., 1:] = mask
        values = values[mask]
        events = events[mask2]

        return RaggedArray(events, mask2.sum(axis=-1)), RaggedArray(values, mask.sum(axis=-1))

    def __get_col_reverse(self, indices, values):
        return (indices[..., -1][:, None] - indices[..., ::-1], values[..., ::-1])

    def to_array(self, ragged=True):
        self._values.ravel()
        self._indices.ravel()
        l = [row.to_array() for row in self]
        if ragged:
            return RaggedArray(l)
        return np.array(l)

    @classmethod
    def from_array(cls, array):
        return cls.from_ragged_array(
            RaggedArray.from_numpy_array(np.asanyarray(array)))

    @classmethod
    def from_ragged_array(cls, ragged_array: RaggedArray) -> 'RunLengthRaggedArray':
        """Construct runlengtharray from a RaggedArray

        Parameters
        ----------
        ragged_array : RaggedArray

        Returns
        -------
        RunLengthRaggedArray
        """
        data = ragged_array.ravel()
        mask = np.insert(data[:-1] != data[1:], 0, True)
        mask[ragged_array._shape.starts] = True
        indices = np.flatnonzero(mask)
        tmp = np.cumsum(unsafe_extend_left(mask))
        row_lens = tmp[ragged_array._shape.ends]-tmp[ragged_array._shape.starts]
        values = data[indices]
        s = np.concatenate([indices, ragged_array._shape.ends])
        s.sort(kind='mergesort')
        indices = RaggedArray(s, row_lens+1)
        start_indices = indices[:, 0][:, np.newaxis]
        indices = indices-start_indices
        return cls(indices,
                   RaggedArray(values, row_lens))# , ragged_array._shape.lengths)

    def max(self, axis=-1, **kwargs):
        assert axis in (-1, 1)
        return self._values.max(axis=-1, **kwargs)

    def argmax(self, axis=-1, **kwargs):
        assert axis in (-1, 1)
        m = self.max(axis=-1, keepdims=True)
        rows, cols = np.nonzero(self._values == m)
        _, idxs = np.unique(rows, return_index=True)
        return self._indices[np.arange(len(idxs)), cols[idxs]]

    def mean(self, axis=-1, **kwargs):
        if axis in (0, -2):
            return self.sum(axis=0)/self.col_counts()
        s = self.sum(axis=-1)
        l = self._row_len
        if self._row_len is None:
            l = self._indices[:, -1]
        return s/l

    def __array_function__(self, func: callable, types: List, args: List, kwargs: Dict):
        if func == np.where:
            condition, x, y = args
            return self.__class__(np.where(x._indices._broadcast_rows(condition), x._indices, y._indices),
                                  np.where(x._values._broadcast_rows(condition), x._values, y._values))
        if func in (np.max, np.amax):
            return func(args[0]._values, *args[1:], **kwargs)
        if func == np.concatenate:
            return rlra_concatenate(*args, **kwargs)
        if func == np.sum:
            return self.sum(*args[1:], **kwargs)
        if func == np.mean:
            return self.mean(*args[1:], **kwargs)
        return NotImplemented

    def col_counts(self):
        indices, counts = np.unique(self.shape[-1], return_counts=True)
        values = len(self)-np.insert(np.cumsum(counts), 0, 0)
        indices = np.insert(indices, 0, 0)
        return RunLengthArray(indices, values[:-1])

    def ravel(self):
        offsets = np.insert(
            np.cumsum(self._indices[:, -1]),
            0, 0)
        indices = (self._indices[:, :-1]+offsets[:-1, np.newaxis]).ravel()
        indices = np.append(indices, offsets[-1])
        values = self._values.ravel()
        return RunLengthArray(indices, values)

    @property
    def size(self) -> int:
        return self.shape[-1].sum()


def rlra_concatenate(rl_ragged_arrays):
    return RunLengthRaggedArray(
        np.concatenate([a._indices for a in rl_ragged_arrays]),
        np.concatenate([a._values for a in rl_ragged_arrays]))
