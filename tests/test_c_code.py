import numpy as np
from numpy.testing import assert_array_equal

from npstructures.copy_segment import compute
print('-----------------------')


def test_copy():
    array_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    array_2 = np.zeros_like(array_1, shape=(5,))
    starts = np.array([1, 5])
    ends = np.array([3, 8])
    compute(array_1, array_2,starts, ends)
    assert_array_equal(array_2, np.array([2, 3, 6, 7, 8]))
