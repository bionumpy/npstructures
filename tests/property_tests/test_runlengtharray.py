from npstructures.runlengtharray import RunLengthArray, RunLength2dArray, RunLengthRaggedArray
from npstructures.mixin import NPSArray
from npstructures import RaggedArray
from numpy import array, int8, int16
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from npstructures.testing import assert_raggedarray_equal
from .strategies import arrays, array_shapes, nested_lists, list_of_arrays, vector_and_indexes, vector_and_startends
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


@given(list_of_arrays(1, 1))
@example(lists=[[0], [0]])
def test_run_length_ragged_array(lists):
    ragged_array = RaggedArray(lists)
    rlarray = RunLengthRaggedArray.from_ragged_array(ragged_array)
    new_array = rlarray.to_array()
    assert_raggedarray_equal(ragged_array, new_array)


@given(vector_and_indexes())
@example(data=(array([0], dtype=int8), (slice(-1, None, None),)))
@example(data=(array([0, 1], dtype=int8), (slice(None, None, 2),)))
@example(data=(array([0, 0, 0], dtype=int8), (slice(None, None, 2),)))
@example(data=(array([0], dtype=int8), (slice(None, 0, None),)))
@example(data=(array([0, 0], dtype=int8), (slice(1, None, -1),)))
@example(data=(array([0, 1], dtype=int16), (-2,)))
@example(data=(array([0], dtype=int8), Ellipsis))
@example(data=(array([0, 1, 1, 1, 1], dtype=int8), (slice(1, 0, None),)))
def test_run_length_indexing(data):
    vector, idx = data
    rla = RunLengthArray.from_array(vector)
    subset = rla[idx]
    if isinstance(subset, RunLengthArray):
        subset = subset.to_array()
    assert_array_equal(subset, vector[idx])


@given(vector_and_startends())
@example(data=(array([0, 0], dtype=int8), [0], [1]))
def test_ragged_slice(data):
    vector, starts, ends = (np.asanyarray(e) for e in data)
    starts = np.minimum(starts, ends)
    ends = np.maximum(starts, ends+1)
    subset = vector.view(NPSArray)[starts:ends]
    assert_raggedarray_equal(subset, RaggedArray([vector[start:end] for start, end in zip(starts, ends)]))


@given(vector_and_startends())
@example(data=(array([0, 0], dtype=int8), [0], [1]))
@example(data=(array([0], dtype=int8), [0, 0], [0, 0]))
def test_ragged_runlength_slice(data):
    vector, starts, ends = (np.asanyarray(e) for e in data)
    rla = RunLengthArray.from_array(vector)
    starts = np.minimum(starts, ends)
    ends = np.maximum(starts, ends+1)
    subset = rla[starts:ends]
    print(subset.to_array())
    assert_raggedarray_equal([row.to_array() for row in subset],
                             [vector[start:end] for start, end in zip(starts, ends)])
