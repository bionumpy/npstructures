import numpy as np
import npstructures as nps
from npstructures import RaggedArray


# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

class TimeRaggedArraySuite:
    def setup(self):
        np.random.seed(1)
        self.matrix = np.random.randint(0, 1000, (1000, 1000), dtype=int)
        self.matrix2 = np.random.randint(0, 1000, (500, 1000), dtype=int)
        self.ragged_array = RaggedArray.from_numpy_array(self.matrix)
        self.ragged_array2 = RaggedArray.from_numpy_array(self.matrix2)
        self.indexes = np.random.randint(0, 1000, 1000)

    def time_nps_row_sum(self):
        self.ragged_array.sum(axis=-1)

    def time_np_row_sum(self):
        self.matrix.sum(axis=-1)

    def time_nps_col_sum(self):
        self.ragged_array.sum(axis=0)

    def time_np_col_sum(self):
        self.matrix.sum(axis=0)

    def time_nps_row_mean(self):
        self.ragged_array.mean(axis=-1)

    def time_np_row_mean(self):
        self.matrix.mean(axis=-1)

    def time_nps_row_std(self):
        self.ragged_array.std(axis=-1)

    def time_np_row_std(self):
        self.matrix.std(axis=-1)

    def time_nps_square(self):
        self.ragged_array ** 2

    def time_np_square(self):
        self.matrix ** 2

    def time_concatenate_nps(self):
        np.concatenate((self.ragged_array, self.ragged_array2), axis=0)

    def time_concatenate_np(self):
        np.concatenate((self.matrix, self.matrix2), axis=0)

    # indexing uses .ravel() for bnp and .flatten() for np
    # since nothing is done just when indexing
    def time_nps_getitem_row(self):
        return self.ragged_array[self.indexes].ravel()

    def time_np_getitem_row(self):
        return self.matrix[self.indexes].flatten()

    def time_nps_getitem_col(self):
        return self.ragged_array[:, 1:-1 ].ravel()

    def time_np_getitem_col(self):
        return self.matrix[:, 1:-1].flatten()


