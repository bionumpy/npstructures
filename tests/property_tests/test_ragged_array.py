import numpy as np
from numpy import array, int8, int16, int32, float16, float32, float64, mean, std
import pytest
from numpy.testing import assert_equal, assert_allclose
from npstructures import RaggedArray
from hypothesis import given, example
from numbers import Number
import hypothesis.extra.numpy as stnp
from .strategies import (
    matrix_and_indexes,
    matrix_and_indexes_and_values,
    matrix_and_integer_array_indexes,
    nested_list_and_indices,
    nested_list_and_slices,
    matrices, nested_lists, arrays, array_shapes,
    two_nested_lists,
    list_of_arrays,
    nonempty_list_of_arrays,
    integers,
    two_arrays,
    array_and_column
)


from npstructures.arrayfunctions import ROW_OPERATIONS

row_operation_functions = [getattr(np, name) for name in ROW_OPERATIONS] + [np.diff]

ufuncs = [np.add, np.subtract, np.multiply, np.bitwise_and, np.bitwise_or, np.bitwise_xor]


@given(matrices())
def test_create_from_numpy_matrix(matrix):
    ra = RaggedArray.from_numpy_array(matrix)
    assert_equal(ra.to_numpy_array(), matrix)


@given(nested_lists())
def test_create_from_nested_list(nested_list):
    ra = RaggedArray(nested_list)
    assert_equal(ra.tolist(), nested_list)


@given(matrix_and_indexes() | matrix_and_integer_array_indexes())
# @given(matrix_and_integer_array_indexes())
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
@example((np.empty(shape=(2, 0), dtype=np.int8),(slice(None, None, None), slice(None, None, None))))
def test_getitem(data):
    array, indices = data
    ra = RaggedArray.from_numpy_array(array)
    result = ra[indices]
    if isinstance(result, RaggedArray):
        result = result.to_numpy_array()
    true_result = array[indices]
    if len(true_result.shape) < 2 or true_result.shape[0] != 0 or true_result.shape[1] == 0:
        assert_equal(result, true_result)


@given(nested_list_and_indices())
@example(([[0], [0, 0]], np.array([1]), [1]))
def test_getitem_ragged(data):
    lists, row_indices, col_indices = data
    ra = RaggedArray(lists)
    res = ra[row_indices, col_indices]
    true = [lists[i][j] for i, j in zip(row_indices, col_indices)]
    assert_equal(res, true)


@given(nested_list_and_slices())
@example(([[0, 0, 0], [0]], 1, slice(-2, None, -1)))
@example(([[0], [0, 0, 0]], 0, slice(-2, -3, -1)))
def test_getitem_ragged_sliced(data):
    lists, row_indices, col_indices = data
    assert isinstance(col_indices, slice)
    ra = RaggedArray(lists)
    res = ra[row_indices, col_indices]
    rows = np.array(lists, dtype=object)
    true = rows[row_indices]
    if isinstance(row_indices, Number):
        true = true[col_indices]
    else:
        true = [np.atleast_1d(row)[col_indices] for row in true]
    if isinstance(res, RaggedArray):
        true = RaggedArray(true)
        assert np.all(res == true)
    else:
        assert_equal(res, true)


@given(matrix_and_indexes(arrays(array_shape=array_shapes(1, 2, 2))), integers())
def test_setitem_single_value(data, value):
    array, indices = data
    ra = RaggedArray.from_numpy_array(array)
    ra[indices] = value
    assert_equal(ra[indices].ravel(), array.dtype.type(value))



@given(matrix_and_indexes_and_values(arrays(array_shape=array_shapes(1, 2, 2))))
@example(data=(array([[0, 0]], dtype=int8),
               array([True]),
               array([[0, 0]], dtype=int8)))
def test_setitem(data):
    array, indices, values = data
    ra = RaggedArray.from_numpy_array(array.copy())
    ra[indices] = values
    array[indices] = values
    assert_equal(ra.to_numpy_array(), array)



@pytest.mark.parametrize("axis", [None, -1])
@pytest.mark.parametrize("function", row_operation_functions + [np.cumsum, np.cumprod])
@given(nested_array_list=list_of_arrays(min_size=1))
@example(nested_array_list=[array([1], dtype=int8),
                            array([0, 0, 0, 0, 0], dtype=int8)],
         function=np.std,
         axis=-1)
@example(nested_array_list=[array([151060739]), array([0, 0, 0])],
         function=np.std,
         axis=-1)
@example(
    nested_array_list=[array([0., 0.00976], dtype=np.float32)],
    function=np.std,
    axis=-1)
@example(nested_array_list=[array([0., 1., 1.], dtype=np.float32)],
         function=np.mean,
         axis=-1)
@example(nested_array_list=[array([], dtype=float32), array([1.], dtype=float32)],
         function=np.cumsum,
         axis=-1)
@example(nested_array_list=[array([6777218.], dtype=float32),
                            array([9999999.], dtype=float32)],
         function=np.cumsum,
         axis=-1,)
def test_array_function(nested_array_list, function, axis):
    ra = RaggedArray(nested_array_list)

    if function in [np.argmin, np.argmax, np.amin, np.amax]:
        if (axis == -1 and any(len(row) == 0 for row in ra)) or \
           (axis is None and len(ra.ravel()) == 0):
            # don't test argmin/argmax on empty rows
            return

    if axis == -1:
        if function == np.cumsum:
            true = RaggedArray([function(row) for row in nested_array_list])
        else:
            true = np.array([function(row) for row in nested_array_list])
    else:
        true = function(np.concatenate(nested_array_list))

    try:
        result = function(ra, axis=axis)
    except TypeError:
        return  # type error is okay, thrown when axis not supported ??

    if isinstance(result, RaggedArray):
        for ragged_row, np_row in zip(result, true):
            assert_allclose(ragged_row, np_row, equal_nan=True, atol=10**(-8))
        # assert all([np.allclose(ragged_row, np_row, equal_nan=True)
        #            for ragged_row, np_row in zip(result, true)])
    else:
        assert_allclose(result, true, equal_nan=True, rtol=10**-6, atol=10**(-8))


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
def test_nonzero_matrix(matrix):
    ra = RaggedArray(matrix)
    np_matrix = np.array(matrix)
    assert_equal(np.nonzero(np_matrix), np.nonzero(ra))


@given(nested_lists())
def test_nonzero(nl):
    ra = RaggedArray(nl)
    nz_row_indices = []
    nz_col_indices = []
    for i, row in enumerate(ra):
        for j, el in enumerate(row):
            if el != 0:
                nz_row_indices.append(i)
                nz_col_indices.append(j)
    assert_equal(np.nonzero(ra), (np.array(nz_row_indices, dtype=int),
                                  np.array(nz_col_indices, dtype=int)))


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


@given(nested_list=list_of_arrays(min_size=1), fill_value=integers())
def test_fill(nested_list, fill_value):
    ra = RaggedArray(nested_list)
    ra.fill(fill_value)
    for row in nested_list:
        row.fill(fill_value)

    assert_equal(ra, RaggedArray(nested_list))


@pytest.mark.parametrize("func", ufuncs)
@given(arrays=two_arrays())
def test_ufuncs(func, arrays):
    array_a, array_b = arrays
    ra_a = RaggedArray.from_numpy_array(array_a)
    ra_b = RaggedArray.from_numpy_array(array_b)
    ra_c = ra_a + ra_b
    assert_equal(array_a+array_b, ra_c.to_numpy_array())


@pytest.mark.parametrize("func", ufuncs)
@given(arrays=array_and_column())
@example(arrays=(array([[0]], dtype=int8), array([[128]], dtype=int16)))
@example(arrays=(array([[-1]], dtype=int16), array([[-32768]], dtype=int16)))
@example(arrays=(array([[0],
                        [0]]),
                 array([[9.0072e+15],
                        [1.0000e+00]], dtype=np.float32)), func=np.add)
@example(arrays=(
    array([[0.],
           [0.]], dtype=np.float16),
    array([[-1.000000e+00],
           [8.388609e+06]], dtype=np.float32)), func=np.add)
@example(arrays=(array([[0],
            [0],
            [0]], dtype=int8), array([[ 8.388609e+06],
            [-1.000000e+00],
            [ 0.000000e+00]], dtype=np.float32)), func=np.bitwise_and)
def test_broadcasting(func, arrays):
    array_a, array_b = arrays
    ra_a = RaggedArray.from_numpy_array(array_a)
    ra_b = array_b
    ra_c = ra_a + ra_b
    if np.issubdtype(array_a.dtype, np.floating) or np.issubdtype(array_b.dtype, np.floating):
        assert_allclose(array_a+array_b, ra_c.to_numpy_array(), rtol=10e-5)
    else:
        assert_equal(array_a+array_b, ra_c.to_numpy_array())


@pytest.mark.parametrize("func", [np.add, np.bitwise_xor])
@given(array_list=list_of_arrays(min_size=1, min_length=1, dtypes=stnp.integer_dtypes()))
@example(array_list=[array([1], dtype=int8)], func=np.subtract)
def test_reductions(array_list, func):
    true = np.array([func.reduce(row) for row in array_list])
    r = func.reduce(RaggedArray(array_list), axis=-1)
    assert r is not NotImplemented
    assert_equal(true, r)
    
        
