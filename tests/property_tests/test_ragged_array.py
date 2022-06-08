import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pytest
from numpy.testing import assert_equal
from npstructures import RaggedArray
import hypothesis.extra.numpy as stnp
from hypothesis import given
from hypothesis.strategies import composite
from strategies import matrix_and_indexes, matrix, nested_list


@given(matrix())
def test_create_from_numpy_matrix(matrix):
    ra = RaggedArray.from_numpy_array(matrix)


@given(nested_list())
def test_create_from_nested_list(nested_list):
    ra = RaggedArray(nested_list)
    assert_equal(ra.tolist(), nested_list)


@given(matrix_and_indexes())
def test_getitem_square_ragged_array(data):
    array, indices = data
    ra = RaggedArray.from_numpy_array(array)

    result = ra[indices]
    if isinstance(result, RaggedArray):
        result = result.to_numpy_array()

    assert_equal(result, array[indices])

