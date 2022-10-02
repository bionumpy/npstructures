from npstructures.runlengtharray import RunLengthArray
from numpy.testing import assert_array_equal
import pytest


@pytest.fixture
def array():
    return [0, 1, 1, 2, 3, 3, 3]


def test_run_length_array(array):
    rlarray = RunLengthArray.from_array(array)
    new_array = rlarray.to_array()
    assert_array_equal(array, new_array)
