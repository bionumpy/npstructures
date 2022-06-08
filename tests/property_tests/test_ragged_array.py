import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pytest
from numpy.testing import assert_equal
from npstructures import RaggedArray
import hypothesis.extra.numpy as stnp
from hypothesis import given
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
def test_getitem(data):
    array, indices = data
    ra = RaggedArray.from_numpy_array(array)

    result = ra[indices]
    if isinstance(result, RaggedArray):
        result = result.to_numpy_array()

    assert_equal(result, array[indices])


@given(matrix_and_indexes(), st.integers())
def test_setitem_single_value(data, value):
    array, indices = data
    ra = RaggedArray.from_numpy_array(array)
    ra[indices] = value
    print()
    print(data, value)
    print(ra, value)
    assert np.all(ra[indices] == value)


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


