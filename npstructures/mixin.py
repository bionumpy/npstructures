import numpy as np
from .raggedarray.raggedslice import ragged_slice


class NPSIndexable:
    def __getitem__(self, idx):
        if hasattr(idx, "start") and hasattr(idx, "stop"):
            if hasattr(idx.start, "__len__") and hasattr(idx.stop, "__len__"):
                return self._ragged_slice(idx.start, idx.stop)
        if hasattr(super(), "__getitem__"):
            return super().__getitem__(idx)
        else:
            raise ValueError(f"Invalid key for {self.__class__.__name__}: {idx}")


class NPSArray(NPSIndexable, np.ndarray):

    __name__ = "array"
    __qualname__ = "array"
    
    def __repr__(self):
        return super().__repr__().replace("NPSArray", "array")

    def _ragged_slice(self, start, stop):
        return ragged_slice(self, start, stop)
