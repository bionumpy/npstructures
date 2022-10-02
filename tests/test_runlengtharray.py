from npstructures.runlengtharray import RunLengthArray
from numpy.testing import assert_array_equal
import numpy as np
import pytest

arrays = [[0, 1, 1, 2, 3, 3, 3],
          [0, 20, 1, 1, 0, 3, 3],
          [2, 2, 2, 1],
          [1],
          [2.3, 19, 32.7, 32.7],
          [0,1,1,2,2,2,3,3,3,3,2,2,2,1,1,0]]

arrays_2 = [[0, 1, 1, 2, 3, 3, 3],
            [10, 1, 1, 2, 3, 2, 2],
            [13, 13, 3, 0],
            [10],
            [32.3, 198, 932.2, 329.7]]


@pytest.mark.parametrize("array", arrays)
def test_run_length_array(array):
    rlarray = RunLengthArray.from_array(array)
    new_array = rlarray.to_array()
    assert_array_equal(array, new_array)


@pytest.mark.parametrize("array1,array2", tuple(zip(arrays, arrays_2)))
def test_add_run_length_array(array1, array2):
    array1, array2 = (np.asanyarray(a) for a in (array1, array2))
    rl_result = RunLengthArray.from_array(array1)+RunLengthArray.from_array(array2)
    assert_array_equal(rl_result.to_array(), array1+array2)


@pytest.mark.parametrize("array", arrays)
def test_getitem_int(array):
    rlarray = RunLengthArray.from_array(array)
    for i, elem in enumerate(array):
        assert rlarray[i] == elem


@pytest.mark.parametrize("array", arrays)
def test_getitem_slice(array):
    rlarray = RunLengthArray.from_array(array)
    for start, _ in enumerate(array):
        for end in range(start+1, len(array)):
            assert_array_equal(rlarray[start:end].to_array(), array[start:end])
