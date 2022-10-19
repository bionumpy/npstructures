from npstructures.runlengtharray import RunLengthArray, RunLength2dArray, RunLengthRaggedArray
from npstructures import RaggedArray
from numpy import array, int8
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from npstructures.testing import assert_raggedarray_equal
from .strategies import arrays, array_shapes, nested_lists, list_of_arrays
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
    rlarray = RunLength2dArray.from_array(np_array)
    new_array = rlarray.to_array()
    assert_array_almost_equal(np_array, new_array)


@given(list_of_arrays(1, 1)) #nested_lists(min_size=1))<
@example(lists=[[0], [0]])
def test_run_length_ragged_array(lists):
    ragged_array = RaggedArray(lists)
    rlarray = RunLengthRaggedArray.from_ragged_array(ragged_array)
    new_array = rlarray.to_array()
    assert_raggedarray_equal(ragged_array, new_array)
