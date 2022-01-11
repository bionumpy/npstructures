import numpy as np


class NumpyList:
    """
    Simple List-like data structure backed by a numpy array
    """

    def __init__(self, dtype=None):
        self._data = None
        self._dtype = dtype
        self._n_elements = 0

    def _initialize_array(self, first_element):
        if self._dtype is None:
            self._dtype = type(first_element)
        self._data = np.zeros(100, dtype=self._dtype)

    def append(self, element):
        if self._data is None:
            self._initialize_array(element)

        # extend data array if necessary
        if self._n_elements == len(self._data):
            new_data = np.zeros(len(self._data)*2, dtype=self._data.dtype)
            new_data[0:len(self._data)] = self._data
            self._data = new_data

        self._data[self._n_elements] = element
        self._n_elements += 1

    def get_nparray(self):
        return self._data[0:self._n_elements]