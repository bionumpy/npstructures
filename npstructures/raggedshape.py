import numpy as np

HANDLED_FUNCTIONS = {}

def implements(np_function):
   "Register an __array_function__ implementation for DiagonalArray objects."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator


class RaggedShape:
    def __init__(self, offsets):
        self._offsets = np.asanyarray(offsets)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def ravel_multi_index(self, indices):
        return self._offsets[indices[0]]+indices[1]

    def unravel_multi_index(self, flat_indices):
        rows = np.searchsorted(self._offsets, flat_indices, side="right")-1
        cols = flat_indices-self._offsets[rows]
        return rows, cols

    def index_array(self):
        diffs = np.zeros(self._offsets[-1], dtype=int)
        diffs[self._offsets[1:-1]] = 1
        return np.cumsum(diffs)

    def __eq__(self, other):
        return np.all(self._offsets == other._offsets)

    @property
    def lengths(self):
        return self.ends-self.starts

    @property
    def starts(self):
        return self._offsets[:-1]

    @property
    def ends(self):
        return self._offsets[1:]

    @property
    def size(self):
        return self._offsets[-1]

    @classmethod
    def asanyshape(cls, shape):
        if isinstance(shape, RaggedShape):
            return shape
        return cls(shape)

    def __getitem__(self, index):
        if isinstance(index, slice):
            new_offsets = self._offsets[index.start:index.stop+1]-self._offsets[index.start]
            return RaggedShape(new_offsets)
        return NotImplemented
