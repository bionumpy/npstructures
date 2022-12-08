import numpy as np
from ..raggedshape import RaggedShape, RaggedView2, RaggedView


class RaggedBase:
    def __init__(self, data, shape):
        self.__data = data
        self._shape = shape
        self.is_contigous = True

    @property
    def size(self):
        return self.__data.size

    @property
    def dtype(self):
        return self.__data.dtype

    def _change_view(self, new_view):
        ret = self.__class__(self.__data, new_view)
        ret.is_contigous = False
        return ret

    def _flatten_myself(self):
        if isinstance(self._shape, (RaggedView2, RaggedView)):
            idx, shape = self._shape.get_flat_indices()
            if np.issubdtype(idx.dtype, bool):
                self.__data = self.__data[:idx.size]
            self.__data = self.__data[idx]
            self._shape = shape
            self.is_contigous = True
        else:
            assert False, self._shape

    def ravel(self):
        if not self.is_contigous:
            self._flatten_myself()
        return self.__data

    def _get_data_range(self, idx):
        if isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, bool):
            return self.__data[:idx.size][idx]
        return self.__data[idx]

    def _set_data_range(self, idx, data):
        if isinstance(idx, np.ndarray) and np.issubdtype(idx.dtype, bool):
            self.__data[:idx.size][idx] = data
        else:
            self.__data[idx] = data
