from npstructures.runlengtharray import RunLengthArray, RunLength2dArray
from numpy import array, int8
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .strategies import arrays, array_shapes
from hypothesis import given, example


@given(arrays(array_shape=array_shapes(1, 1, 1)))
def test_run_length_array(np_array):
    rlarray = RunLengthArray.from_array(np_array)
    new_array = rlarray.to_array()
    assert_array_equal(np_array, new_array)


# @pytest.mark.skip("unimplemented")
@given(arrays(array_shape=array_shapes(1, 2, 2)))
@example(array([[0]], dtype=int8))
@example(np_array=array([[0], [0]], dtype=int8))
def test_run_length_2d_array(np_array):
    print(np_array)
    print("------------------------------")
    rlarray = RunLength2dArray.from_array(np_array)
    new_array = rlarray.to_array()
    assert_array_almost_equal(np_array, new_array)
