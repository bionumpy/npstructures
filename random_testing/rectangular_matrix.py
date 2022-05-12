import numpy as np
from numpy.random import randint, rand
from npstructures.raggedarray import RaggedArray


def ragged_from_array(array):
    n_rows, n_cols = array.shape
    return RaggedArray(array, np.full(n_rows, n_cols))


def ragged_to_array(ragged):
    n_rows = ragged.shape.n_rows()
    n_cols = ragged.shape.lengths[0]
    assert np.all(ragged.shape.lengths == n_cols)
    return ragged.ravel().reshape(n_rows, n_cols)


class IndexGenerator:
    def __init__(self, array):
        self.array = array
        self.n_row, self.n_col = array.shape

    def row_indices(self, max_n=100):
        size = randint(0, max_n, size=1)
        return randint(0, self.n_row, size)

    def col_indices(self, max_n=100):
        size = randint(0, max_n, size=1)
        return (slice(0), randint(0, self.n_col, size))


def generate_array(min_col=1, max_col=100, min_row=1, max_row=100):
    shape = (randint(min_row, max_row), randint(min_col, max_col))
    return rand(shape[0]*shape[1]).reshape(shape)


def main():
    array = generate_array()
    index_generator = IndexGenerator(array)
    for indices in (index_generator.row_indices(), index_generator.col_indices()):
        print(indices)
        assert np.all(array[indices] == RaggedArray.from_numpy_array(array)[indices].to_numpy_array())


main()
