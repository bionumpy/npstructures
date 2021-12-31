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

    @property
    def lengths(self):
        return self.ends-self.starts

    @property
    def starts(self):
        return self._offsets[:-1]

    @property
    def ends(self):
        return self._offsets[1:]]

    @property
    def size(self):
        return self._offsets[-1]
