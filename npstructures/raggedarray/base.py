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

    @property
    def size(self) -> int:
        return self.__data.size

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.__data.dtype

    def _change_view(self, new_view):
        ret = self.__class__(self.__data, new_view)
        #ret.is_contigous = False
        return ret

    def _flatten_myself(self):
        if not isinstance(self._shape, (RaggedView2, RaggedView)):
            return
        if isinstance(self._shape, RaggedView2) and self._shape.n_rows>0 and any(self.dtype==d for d in (np.uint8, np.int32, np.int64, np.uint64, np.float64)):
            shape = self._shape.get_shape()
            if self.__data.size == 0:
                pass
            elif self._shape.col_step == 1:
                self.__data = c_extract_segments(self.__data, self._shape, shape, self._shape.col_step)
            else:
                self.__data = native_extract_segments(self.__data, self._shape, shape, self._shape.col_step)
            self._shape = shape
            return

        idx, shape = self._shape.get_flat_indices()
        #if np.issubdtype(idx.dtype, bool):
        #    self.__data = self.__data[:idx.size]
        self.__data = self.__data[idx]
        self._shape = shape
        self.is_contigous = True
        # else:
        #    #assert False, self._shape

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
