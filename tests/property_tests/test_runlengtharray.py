from npstructures.runlengtharray import RunLengthArray# , RunLength2dArray
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .strategies import arrays, array_shapes
from hypothesis import given


@given(arrays(array_shape=array_shapes(1, 1, 1)))
def test_run_length_array(array):
    rlarray = RunLengthArray.from_array(array)
    new_array = rlarray.to_array()
    assert_array_equal(array, new_array)


@pytest.mark.skip("unimplemented")
@given(arrays(array_shape=array_shapes(1, 2, 2)))
def test_run_length_2d_array(array):
    rlarray = RunLength2dArray.from_array(array)
    new_array = rlarray.to_array()
    assert_array_almost_equal(array, new_array)
