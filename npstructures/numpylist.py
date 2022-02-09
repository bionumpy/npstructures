import numpy as np


class NumpyList:
    """
    Simple List-like data structure backed by a numpy array
    """

    def __init__(self, dtype=None):
        self._data = np.empty(0)
        if dtype is not None:
            self._data = self._data.astype(dtype)
        self._dtype = dtype
        self._n_elements = 0

    def _initialize_array(self, first_element):
        if self._dtype is None:
            self._dtype = type(first_element)
            self._data = self._data.astype(self._dtype)
        self._data = np.zeros(100, dtype=self._dtype)

    def _extend_data(self, new_length):
        new_data = np.zeros(new_length, dtype=self._data.dtype)
        new_data[0:len(self._data)] = self._data
        self._data = new_data

    def append(self, element):
        if len(self._data) == 0:
            self._initialize_array(element)

        # extend data array if necessary
        if self._n_elements == len(self._data):
           self._extend_data(len(self._data)*2)

        self._data[self._n_elements] = element
        self._n_elements += 1

    def extend(self, elements):
        if self._data is None:
            self._initialize_array(elements[0])
        if self._n_elements + len(elements) >= len(self._data):
            self._extend_data((self._n_elements+len(elements))*2)
        self._data[self._n_elements:self._n_elements+len(elements)] = elements
        self._n_elements += len(elements)

    def __getitem__(self, item):
        return self.get_nparray()[item]  # works for when item is negative, etc

    def get_nparray(self):
        return self._data[0:self._n_elements]

    def set_n_elements(self, n):
        self._n_elements = n

    def copy(self):
        new = NumpyList(dtype=self._dtype)
        new.extend(self.get_nparray())
        return new

    def __eq__(self, other):
        return np.all(self.get_nparray() == other.get_nparray())

    def __len__(self):
        return self._n_elements

    def __str__(self):
        return str(self.get_nparray())

    def __repr__(self):
        return "NumpyList(" + str(self) + ")"
