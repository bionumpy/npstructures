import numpy as np


class RaggedBase:
    def __init__(self, data, shape):
        self.__data = data
        self._shape = shape

    @property
    def size(self):
        return self.__data.size

    @property
    def dtype(self):
        return self.__data.dtype

    def ravel(self):
        return self.__data
