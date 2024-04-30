import numpy as np

import npstructures as nps
from npstructures import RaggedArray


# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for key in self.d.keys():
            pass

    def time_values(self):
        for value in self.d.values():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256


class TimeRaggedArraySuite:
    def setup(self):
        np.random.seed(1)
        self.matrix = np.random.randint(0, 1000, (1000, 1000), dtype=int)
        self.ragged_array = RaggedArray.from_numpy_array(self.matrix)
        self.indexes = np.random.randint(0, 1000, 1000)

    def time_nps_row_sum(self):
        self.ragged_array.sum(axis=-1)

    def time_np_row_sum(self):
        self.matrix.sum(axis=-1)

    def time_nps_square(self):
        self.ragged_array ** 2

    def time_np_square(self):
        self.matrix ** 2

    def time_nps_getitem_row(self):
        self.ragged_array[self.indexes]

    def time_np_getitem_row(self):
        self.matrix[self.indexes]

    def time_nps_getitem_col(self):
        self.ragged_array[:, self.indexes]

    def time_np_getitem_col(self):
        self.matrix[:, self.indexes]


