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
    
def test_add_scalar(array_list):
    ra = RaggedArray.from_array_list(array_list)
    result = np.add(ra, 1)
    true = RaggedArray.from_array_list([[e+1 for e in row] for row in array_list])
    assert result.equals(true)

def test_add_array(array_list):
    ra = RaggedArray.from_array_list(array_list)
    adds = np.arange(4)
    result = np.add(ra, adds[:, None])
    true = RaggedArray.from_array_list([[e+i for e in row] for i, row in enumerate(array_list)])
    assert result.equals(true)

def test_add_ra(array_list):
    ra = RaggedArray.from_array_list(array_list)
    adds = np.arange(4)
    result = np.add(ra, ra)
    true = RaggedArray.from_array_list([[e*2 for e in row] for row in array_list])
    assert result.equals(true)

def test_add_operator(array_list):
    ra = RaggedArray.from_array_list(array_list)
    adds = np.arange(4)
    result = ra+ra
    true = RaggedArray.from_array_list([[e*2 for e in row] for row in array_list])
    assert result.equals(true)

def test_sum(array_list):
    ra = RaggedArray.from_array_list(array_list)
    assert ra.sum() == sum(e for row in array_list for e in row)

def test_rowsum(array_list):
    ra = RaggedArray.from_array_list(array_list)
    s = ra.sum(axis=-1)
    true = np.array([sum(row) for row in array_list])
    assert np.all(s == true)

def test_rowmean(array_list):
    ra = RaggedArray.from_array_list(array_list)
    m = ra.mean(axis=-1)
    true = np.array([sum(row)/len(row) for row in array_list])
    assert np.all(m == true)

def test_concatenate(array_list):
    ra = RaggedArray.from_array_list(array_list)
    cat = RaggedArray.concatenate([ra, ra])
    true = RaggedArray.from_array_list(array_list+array_list)
    assert cat.equals(true)
