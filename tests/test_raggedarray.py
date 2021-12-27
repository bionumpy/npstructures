import pytest
import numpy as np
from npstructures import RaggedArray

@pytest.fixture
def array_list():
    return [[0, 1, 2],
            [2, 1],
            [1, 2, 3, 4],
            [3]]

def test_getitem_tuple(array_list):
    ra = RaggedArray.from_array_list(array_list)
    assert ra[2, 1] == 2
    
def test_getitem_int(array_list):
    ra = RaggedArray.from_array_list(array_list)
    assert np.all(ra[1] == [2, 1])

def test_getitem_slice(array_list):
    ra = RaggedArray.from_array_list(array_list)
    subset = ra[1:3]
    true = RaggedArray.from_array_list(array_list[1:3])
    assert subset.equals(true)

def test_getitem_list(array_list):
    ra = RaggedArray.from_array_list(array_list)
    subset = ra[[0, 2]]
    true = RaggedArray.from_array_list([array_list[0], array_list[2]])
    assert subset.equals(true)

def test_getitem_boolean(array_list):
    ra = RaggedArray.from_array_list(array_list)
    subset = ra[np.array([True, False, False, True])]
    true = RaggedArray.from_array_list([array_list[0], array_list[3]])
    assert subset.equals(true)
    
