import pytest
from npstructures.runlengtharray import RunLengthArray, RunLength2dArray, RunLengthRaggedArray
from npstructures.mixin import NPSArray
from npstructures import RaggedArray
from numpy import array, int8, int16
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from npstructures.testing import assert_raggedarray_equal
from .strategies import arrays, array_shapes, nested_lists, list_of_arrays, vector_and_indexes, vector_and_startends, two_arrays, matrix_and_row_indexes, array_and_column#, matrix_and_integer_array_indexes
from hypothesis import given, example
import hypothesis.extra.numpy as stnp
ufuncs = [np.add, np.subtract, np.multiply, np.bitwise_and, np.bitwise_or, np.bitwise_xor]


@given(arrays(array_shape=array_shapes(1, 1, 1)))
def test_run_length_array(np_array):
    rlarray = RunLengthArray.from_array(np_array)
    print(rlarray._values, rlarray._events)
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
#@example(arrays=(array([0, 1], dtype=int16), array([0, 0], dtype=int8)), func=np.subtract)
def test_ufuncs_integers_runlength_vector(func, arrays):
    array_a, array_b = arrays
    ra_a = RunLengthArray.from_array(array_a)
    ra_b = RunLengthArray.from_array(array_b)
    ra_c = func(ra_a, ra_b)
    assert_array_equal(func(array_a, array_b), ra_c.to_array())


@pytest.mark.parametrize("func", [np.add, np.multiply, np.subtract, np.bitwise_and, np.bitwise_or, np.bitwise_xor])
@given(arrays=array_and_column(dtype=stnp.integer_dtypes(), array_shape=array_shapes(1, 2, 2)))
def test_ufuncs_integers_runlength_matrix(func, arrays):
    array_a, array_b = arrays
    ra_a = RunLength2dArray.from_array(array_a)
    ra_b = array_b
    ra_c = func(ra_a, ra_b)
    assert_array_equal(func(array_a, array_b), ra_c.to_array())



@pytest.mark.parametrize("func", [np.sum, np.any, np.all])
@given(arrays(array_shape=array_shapes(1, 1, 1)))
#@example(array=array([0], dtype=int8), func=all)
#@example(array=array([1, -4785074604081153, -4785074604081153,
#                      -4785074604081153]), func=np.mean)
def test_reductions(func, array):
    rla = RunLengthArray.from_array(array)
    assert_array_almost_equal(func(rla), func(array))


@pytest.mark.parametrize("func", [np.any, np.all, np.sum])
@given(arrays(array_shape=array_shapes(1, 2, 2)))
#@example(array=array([[0, 0]], dtype=int8), func=np.sum)
#@example(array=array([[0, 2, 2]], dtype=int8), func=np.sum)
def test_2dreductions(func, array):
    rla = RunLength2dArray.from_array(array)
    assert_array_equal(func(rla, axis=-1), func(array, axis=-1))


@pytest.mark.parametrize("func", [np.sum, np.any])
@given(arrays(array_shape=array_shapes(1, 2, 2)))
#@example(array=array([[-65],
#                      [-65]], dtype=int8), func=np.sum)
#@example(array=array([[-25, 103]], dtype=int8), func=np.sum)
#@example(array=array([[-25, 103]], dtype=int8), func=np.any)
def test_col_reductions(func, array):
    rla = RunLength2dArray.from_array(array)
    assert_array_equal(func(rla, axis=0), func(array, axis=0))



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
    assert_raggedarray_equal([row.to_array() for row in subset],
                             [vector[start:end] for start, end in zip(starts, ends)])


@given(matrix_and_row_indexes(matrices=arrays(array_shape=array_shapes(1, 2, 2))))# | matrix_and_integer_array_indexes(matrices=arrays(array_shape=array_shapes(1, 2, 2))))
# @example(data=(array([[0]], dtype=int8), (array([[0, 0]]), array([[0, 0]]))))
# @example(data=(array([[0]], dtype=int8), (Ellipsis, 0, slice(None, 0, None))))
def test_getitem(data):
    array, indices = data
    ra = RunLength2dArray.from_array(array)
    true_result = array[indices]
    if len(true_result.shape) < 2 or true_result.shape[0] != 0 or true_result.shape[1] == 0:
        result = ra[indices]
        if isinstance(result, (RunLength2dArray, RunLengthArray)):
            result = result.to_array()
        assert_array_equal(result, true_result)


@given(vector_and_startends())
@example(data=(array([0], dtype=int8), [0], [0]))
def test_from_intervals(data):
    vec, starts, ends = data
    starts = np.asanyarray(starts)
    ends = np.asanyarray(ends)

    starts = np.minimum(starts, ends)
    ends = np.maximum(starts, ends+1)
    n_intervals = len(starts)
    row_len = len(vec)
    true = np.zeros((n_intervals, row_len), dtype=int)
    for i, (start, end) in enumerate(zip(starts, ends)):
        true[i, start:end] += 1
    result = RunLength2dArray.from_intervals(starts, ends, row_len).to_array()
    assert_array_equal(result, true)


@given(vector_and_startends())
def _test_from_intervals_1d(data):
    vec, starts, ends = data
    starts = np.asanyarray(starts)
    ends = np.asanyarray(ends)

    starts = np.minimum(starts, ends)
    ends = np.maximum(starts, ends+1)
    n_intervals = len(starts)
    row_len = len(vec)
    true = np.zeros((row_len), dtype=bool)
    for (start, end) in enumerate(zip(starts, ends)):
        true[int(start):int(end)] |= True
    result = RunLength2dArray.from_intervals(starts, ends, row_len).to_array()
    assert_array_equal(result, true)


