import pytest
import numpy as np
from npstructures import RaggedArray
from npstructures.indexed_raggedarray import IRaggedArray, IRaggedArrayWithReverse
@pytest.fixture
def array_list():
    return [[0, 1, 2],
            [2, 1],
            [1, 2, 3, 4],
            [3]]

def test_getitem_tuple(array_list):
    ra = RaggedArray(array_list)
    assert ra[2, 1] == 2
    

@pytest.mark.parametrize("cls", [RaggedArray])
def test_getitem_int(array_list, cls):
    ra = cls(array_list)
    assert np.all(ra[1] == [2, 1])
    assert np.all(ra == cls(array_list))

@pytest.mark.parametrize("cls", [RaggedArray])
def test_getitem_slice(array_list, cls):
    ra = cls(array_list)
    subset = ra[1:3]
    true = cls(array_list[1:3])
    assert subset.equals(true)
    assert np.all(ra == cls(array_list))

@pytest.mark.parametrize("cls", [RaggedArray])
def test_getitem_list(array_list, cls):
    ra = cls(array_list)
    subset = ra[[0, 2]]
    true = cls([array_list[0], array_list[2]])
    assert subset.equals(true)
    assert np.all(ra == cls(array_list))

@pytest.mark.parametrize("cls", [RaggedArray])
def test_getitem_boolean(array_list, cls):
    ra = cls(array_list)
    subset = ra[np.array([True, False, False, True])]
    true = cls([array_list[0], array_list[3]])
    assert subset.equals(true)
    assert np.all(ra == cls(array_list))
    
@pytest.mark.parametrize("cls", [RaggedArray])
def test_add_scalar(array_list, cls):
    ra = cls(array_list)
    result = np.add(ra, 1)
    true = cls([[e+1 for e in row] for row in array_list])
    print(ra, true)
    assert result.equals(true)
    assert np.all(ra == cls(array_list))

@pytest.mark.parametrize("cls", [RaggedArray])
def test_add_array(array_list, cls):
    ra = cls(array_list)
    adds = np.arange(4)
    result = np.add(ra, adds[:, None])
    true = cls([[e+i for e in row] for i, row in enumerate(array_list)])

    assert result.equals(true)
    assert np.all(ra == cls(array_list))

@pytest.mark.parametrize("RaggedArray", [RaggedArray])
def test_add_ra(array_list, RaggedArray):
    ra = RaggedArray(array_list)
    adds = np.arange(4)
    result = np.add(ra, ra)
    true = RaggedArray([[e*2 for e in row] for row in array_list])
    assert result.equals(true)
    assert np.all(ra == RaggedArray(array_list))

@pytest.mark.parametrize("RaggedArray", [RaggedArray])
def test_add_operator(array_list, RaggedArray):
    ra = RaggedArray(array_list)
    adds = np.arange(4)
    result = ra+ra
    true = RaggedArray([[e*2 for e in row] for row in array_list])
    assert result.equals(true)
    assert np.all(ra == RaggedArray(array_list))

def test_sum(array_list):
    ra = RaggedArray(array_list)
    assert ra.sum() == sum(e for row in array_list for e in row)
    assert np.all(ra == RaggedArray(array_list))

def test_rowsum(array_list):
    ra = RaggedArray(array_list)
    s = ra.sum(axis=-1)
    true = np.array([sum(row) for row in array_list])
    assert np.all(s == true)
    assert np.all(ra == RaggedArray(array_list))

def test_rowmean(array_list):
    ra = RaggedArray(array_list)
    m = ra.mean(axis=-1)
    true = np.array([sum(row)/len(row) for row in array_list])
    assert np.all(m == true)
    assert np.all(ra == RaggedArray(array_list))

def test_concatenate(array_list):
    ra = RaggedArray(array_list)
    cat = np.concatenate([ra, ra])
    true = RaggedArray(array_list+array_list)
    assert cat.equals(true)
    assert np.all(ra == RaggedArray(array_list))

@pytest.mark.parametrize("RaggedArray", [RaggedArray, IRaggedArrayWithReverse])
def test_nonzero(array_list, RaggedArray):
    ra = RaggedArray(array_list)
    rows, indices = ra.nonzero()
    assert np.all(rows ==    [0, 0, 1, 1, 2, 2, 2, 2, 3])
    assert np.all(indices == [1, 2, 0, 1, 0, 1, 2, 3, 0])
    assert np.all(ra == RaggedArray(array_list))

def test_zeros_like(array_list):
    ra = RaggedArray(array_list)
    new = np.zeros_like(ra)
    assert np.all(new._data == 0)
    assert new.shape == ra.shape
    assert np.all(ra == RaggedArray(array_list))

@pytest.mark.parametrize("cls", [RaggedArray])
def test_setitem_int(array_list, cls):
    ra = cls(array_list)
    ra[1] = 10
    array_list[1] = [10, 10]
    assert np.all(ra == cls(array_list))
    # assert np.all(ra[1] == [2, 1])

@pytest.mark.parametrize("cls", [RaggedArray])
def test_setitem_slice(array_list, cls):
    ra = cls(array_list)
    ra[1:3] = [[10], [20]]
    array_list[1] = [10, 10]
    array_list[2] = [20, 20, 20, 20]
    assert np.all(ra==cls(array_list))

@pytest.mark.parametrize("cls", [RaggedArray])
def test_setitem_list(array_list, cls):
    ra = cls(array_list)
    ra[[0, 2]] = RaggedArray([[10, 10, 10], [20, 20, 20, 20]])
    array_list[0] = [10, 10, 10]
    array_list[2] = [20, 20, 20, 20]
    assert np.all(ra==cls(array_list))

@pytest.mark.parametrize("cls", [RaggedArray])
def test_setitem_boolean(array_list, cls):
    ra = cls(array_list)
    ra[np.array([True, False, False, True])] = 0
    array_list[0] = [0, 0, 0]
    array_list[3] = [0]
    assert np.all(ra==cls(array_list))

