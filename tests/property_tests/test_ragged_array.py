import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pytest
from numpy.testing import assert_equal
from npstructures import RaggedArray
import hypothesis.extra.numpy as stnp
from hypothesis import given, example
from strategies import matrix_and_indexes, matrices, nested_lists, arrays, array_shapes, \
    two_nested_lists, \
    list_of_arrays, \
    nonempty_list_of_arrays
import hypothesis.strategies as st
from npstructures.arrayfunctions import ROW_OPERATIONS

row_operation_functions = [getattr(np, name) for name in ROW_OPERATIONS] + [np.diff]


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


@pytest.mark.parametrize("function", row_operation_functions)
@given(nested_array_list=list_of_arrays())
def test_row_function(nested_array_list, function):
    ra = RaggedArray(nested_array_list)
    result = function(ra, axis=-1)
    true = np.array([function(row) for row in nested_array_list])

    if function in [np.cumsum, np.cumprod, np.diff]:
        assert all([np.allclose(ragged_row, np_row) for ragged_row, np_row in zip(result, true)])
    else:
        assert np.allclose(result, true)
        assert np.all(ra == RaggedArray(nested_array_list))


@given(two_nested_lists())
def test_concatenate(lists):
    list1, list2 = lists
    assert type(list1[0]) == list
    ra1 = RaggedArray(list1)
    ra2 = RaggedArray(list2)
    concatenated_ra = np.concatenate([ra1, ra2])
    concatenated_lists = list1 + list2
    assert_equal(concatenated_ra, RaggedArray(concatenated_lists))


@given(matrices())
def test_nonzero(matrix):
    ra = RaggedArray(matrix)
    np_matrix = np.array(matrix)
    assert_equal(np.nonzero(np_matrix), np.nonzero(ra))


@given(nested_lists())
def test_nonzero(nl):
    ra = RaggedArray(nl)
    nz_row_indices = []
    nz_col_indices = []
    for i,row in enumerate(ra):
        for j,el in enumerate(row):
            if el != 0:
                nz_row_indices.append(i)
                nz_col_indices.append(j)
    assert_equal( np.nonzero(ra), (np.array(nz_row_indices,dtype=int), np.array(nz_col_indices,dtype=int)) )


@pytest.mark.parametrize("axis", [-1, None])
@given(nested_list=nonempty_list_of_arrays())
def test_unique(nested_list, axis):
    ra = RaggedArray(nested_list)
    unique, counts = np.unique(ra, axis=axis, return_counts=True)

    if axis is None:
        true_unique, true_counts = np.unique(np.concatenate(nested_list), return_counts=True)
        assert_equal(unique, true_unique)
        assert_equal(counts, true_counts)
    else:
        true = [np.unique(row, return_counts=True) for row in nested_list]
        true_unique = [t[0] for t in true]
        true_counts = [t[1] for t in true]

        assert_equal(unique.tolist(), true_unique)
        assert_equal(counts.tolist(), true_counts)


@pytest.mark.parametrize("function", [np.zeros_like, np.ones_like, np.empty_like])
@given(nested_list=nonempty_list_of_arrays())
def test_x_like(nested_list, function):
    print(nested_list)
    ra = RaggedArray(nested_list)
    new = function(ra)

    assert_equal(
        new,
        RaggedArray([function(row) for row in nested_list])
    )


@given(nested_lists(min_size=1))
def test_save_load(nested_list):
    ra = RaggedArray(nested_list)
    ra.save("ra.test.npz")
    ra2 = RaggedArray.load("ra.test.npz")
    assert_equal(ra, ra2)
