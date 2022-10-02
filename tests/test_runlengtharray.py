from npstructures.runlengtharray import RunLengthArray
from numpy.testing import assert_array_equal
import numpy as np
import pytest

arrays = [[0, 1, 1, 2, 3, 3, 3],
          [0, 20, 1, 1, 0, 3, 3],
          [2, 2, 2, 1],
          [1],
          [2.3, 19, 32.7, 32.7]]


@pytest.mark.parametrize("array", arrays)
def test_run_length_array(array):
    rlarray = RunLengthArray.from_array(array)
    new_array = rlarray.to_array()
    assert_array_equal(array, new_array)


def test_add_run_length_array():
    array1, array2 = (np.asanyarray(a) for a in arrays[:2])
    rl_result = RunLengthArray.from_array(array1)+RunLengthArray.from_array(array2)
    assert_array_equal(rl_result.to_array(), array1+array2)
