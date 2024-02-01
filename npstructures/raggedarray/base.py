from functools import lru_cache

from ..util import np
import numpy.typing as npt
from ..raggedshape import RaggedShape, RaggedView2, RaggedView, native_extract_segments, c_extract_segments


class RaggedBase:
    '''
    Base class for ragged arrays.
    Handles evertything to do with the underlying data buffer.
    '''

    def __init__(self, data, shape):
        self.__data = data
        self._shape = shape
        if isinstance(shape, (RaggedView, RaggedView2)):
            self.is_contigous = False
        else:
            self.is_contigous = True
        self._size = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__data!r}, {self._shape!r})"

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = int(np.sum(self._shape.lengths))
        return self._size
        return self.__data.size

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.__data.dtype

    @property
    def _cls(self):
        return self.__class__

    def _change_view(self, new_view):
        ret = self._cls(self.__data, new_view)
        return ret

    def _flatten_myself(self):
        #assert not self.is_contigous
        if not isinstance(self._shape, (RaggedView2, RaggedView)):
            return
        # if False and isinstance(self._shape, RaggedView2) and self._shape.n_rows>0 and any(self.dtype==d for d in (np.uint8, np.int32, np.int64, np.uint64, np.float64)):
        #     shape = self._shape.get_shape()
        #     if self.__data.size == 0:
        #         pass
        #     elif self._shape.col_step == 1 and isinstance(self.__data, np.ndarray) and np.issubdtype(self.__data.dtype, np.integer):
        #         self.__data = c_extract_segments(self.__data, self._shape, shape, self._shape.col_step)
        #     else:
        #         self.__data = native_extract_segments(self.__data, self._shape, shape, self._shape.col_step)
        #     self._shape = shape
        #     assert isinstance(shape, RaggedShape)
        #     self.is_contigous = True
        #     return
        idx, shape = self._shape.get_flat_indices()
        self.__data = self.__data[idx]
        self._shape = shape
        self.is_contigous = True

    def ravel(self) -> npt.ArrayLike:
        """Return a flattened view of the data.

        For now it makes the data contigous on this call. Changes the
        state of the array

        Returns
        -------
        npt.ArrayLike
        """

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
