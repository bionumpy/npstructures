import numpy as np
from .raggedarray.raggedslice import ragged_slice


class NPSIndexable:
    def __getitem__(self, idx):
        if hasattr(idx, "start") and hasattr(idx, "stop"):
            if hasattr(idx.start, "__len__") and hasattr(idx.stop, "__len__"):
                return self._ragged_slice(idx.star, idx.stop)
        return super().__getitem__(idx)


class IndexableArray(NPSIndexable, np.ndarray):
    def _ragged_slice(self, start, stop):
        return ragged_slice(self, start, stop)
