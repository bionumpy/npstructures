import numpy as np
import pytest
from numpy.testing import assert_equal
from npstructures import RaggedArray
import hypothesis.extra.numpy as stnp
from hypothesis import given


@pytest.fixture
def array():
    return np.array([[0, 1, 2], [2, 1, 0]])


@pytest.fixture
def indices():
    return [0]


@given(stnp.arrays(float, (2, 3)))
def test_getitem(array, indices):
    ra = RaggedArray.from_numpy_array(array)
    assert_equal(ra[indices].to_numpy_array(),
                 array[indices])
