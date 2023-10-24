import numpy as np
from numpy.testing import assert_array_equal

from npstructures import ragged_slice



def test_ragged_slice():
    data = np.array([0, 4])
    starts = np.array([3])
    ends = np.array([4])
    result = ragged_slice(data, starts, ends)
    assert_array_equal(result, np.array([[]]))

