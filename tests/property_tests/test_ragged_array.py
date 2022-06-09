import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pytest
from numpy.testing import assert_equal
from npstructures import RaggedArray
import hypothesis.extra.numpy as stnp
from hypothesis import given, example
from strategies import matrix_and_indexes, matrices, nested_lists, arrays, array_shapes
import hypothesis.strategies as st
from npstructures.arrayfunctions import ROW_OPERATIONS

row_operation_functions = [getattr(np, name) for name in ROW_OPERATIONS]


@given(matrices())
def test_create_from_numpy_matrix(matrix):
    ra = RaggedArray.from_numpy_array(matrix)
    assert_equal(ra.to_numpy_array(), matrix)


@given(nested_lists())
def test_create_from_nested_list(nested_list):
    ra = RaggedArray(nested_list)
    assert_equal(ra.tolist(), nested_list)


@given(matrix_and_indexes())
@example((np.array([[0]], dtype=np.int16), (slice(None, None, None), 0)))
@example((np.array([[0, 0]], dtype=np.int8), (slice(None, None, None), slice(None, None, 2))))
@example((np.empty(shape=(1, 0), dtype=np.int8), (0, slice(None, None, None))))
@example((np.array([[0]], dtype=np.int8), (0, slice(None, None, -1))))
@example((np.empty(shape=(1, 0), dtype=np.int8), (slice(None, None, 1), Ellipsis)))
@example((np.array([[0, 1]], dtype=np.int8), (0, slice(None, None, -1))))
@example((np.empty(shape=(0, 0), dtype=np.int8), tuple()))
@example((np.array([[0]], dtype=np.int8), (slice(None, 0, None), slice(None, None, 1))))
@example((np.empty(shape=(0, 0), dtype=np.int8), (Ellipsis,)))
@example((np.array([[0, 0]], dtype=np.int8), (0, slice(1, None, -1))))
@example((np.empty(shape=(2, 0), dtype=np.int8), (Ellipsis, 0, slice(None, None, None))))
@example((np.empty(shape=(0, 0), dtype=np.int8), (slice(None, None, None),)))
@example((np.array([[0], [1]], dtype=np.int8), (0, -1)))
@example((np.array([[0, 0, 0]], dtype=np.int32), (0, slice(-1, None, -1))))
@example((np.array([[0, 0, 0]], dtype=np.int32), (0, slice(None, None, 2))))
@example((np.empty(shape=(2, 0), dtype=np.int8), (slice(None, -1, None),)))
@example((np.array([[0, 0]], dtype=np.int8), (0, slice(1, 0, None))))
def test_getitem(data):
    array, indices = data
    ra = RaggedArray.from_numpy_array(array)
    result = ra[indices]
    if isinstance(result, RaggedArray):
        result = result.to_numpy_array()
    true_result = array[indices]
    if len(true_result.shape) < 2 or true_result.shape[0] != 0 or true_result.shape[1] == 0:
        assert_equal(result, true_result)


@given(matrix_and_indexes(arrays(array_shape=array_shapes(1, 2, 2))), st.integers())
def test_setitem_single_value(data, value):
    array, indices = data
    ra = RaggedArray.from_numpy_array(array)
    ra[indices] = value
    assert_equal(ra[indices], array.dtype.type(value))


@given(nested_lists())
def test_zeros_like(array_list):
    ra = RaggedArray(array_list)
    new = np.zeros_like(ra)
    assert_equal(new.ravel(), 0)
    assert_equal(new.shape, ra.shape)


@pytest.mark.parametrize("function", row_operation_functions)
@given(nested_array_list=nested_lists(arrays(stnp.integer_dtypes(), array_shapes(1, 1, 1))))
def test_row_function(nested_array_list, function):
    print(nested_array_list)
    ra = RaggedArray(nested_array_list)
    result = function(ra, axis=-1)
    true = np.array([function(row) for row in nested_array_list])

    if function in [np.cumsum, np.cumprod]:
        assert all(np.allclose(ragged_row, np_row) for ragged_row, np_row in zip(result, true))
    else:
        assert np.allclose(result, true)
        assert np.all(ra == RaggedArray(nested_array_list))
