import pytest
from npstructures.runlengtharray import RunLengthArray, RunLength2dArray, RunLengthRaggedArray
from npstructures.mixin import NPSArray
from npstructures import RaggedArray
from numpy import array, int8, int16
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from npstructures.testing import assert_raggedarray_equal
from .strategies import arrays, array_shapes, nested_lists, list_of_arrays, vector_and_indexes, vector_and_startends, two_arrays
from hypothesis import given, example
import hypothesis.extra.numpy as stnp
ufuncs = [np.add, np.subtract, np.multiply, np.bitwise_and, np.bitwise_or, np.bitwise_xor]

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


@pytest.mark.parametrize("func", [np.add, np.multiply, np.subtract, np.bitwise_and, np.bitwise_or, np.bitwise_xor])
@given(arrays=two_arrays(dtype=stnp.integer_dtypes(), array_shape=array_shapes(1, 1, 1)))
@example(arrays=(array([0, 1], dtype=int16), array([0, 0], dtype=int8)), func=np.subtract)
def test_ufuncs_integers_runlength_vector(func, arrays):
    array_a, array_b = arrays
    ra_a = RunLengthArray.from_array(array_a)
    ra_b = RunLengthArray.from_array(array_b)
    ra_c = func(ra_a, ra_b)
    assert_array_equal(func(array_a, array_b), ra_c.to_array())




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
