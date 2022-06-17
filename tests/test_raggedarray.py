import pytest
import numpy as np
from npstructures import RaggedArray


@pytest.fixture
def array_list():
    return [[0, 1, 2], [2, 1], [1, 2, 3, 4], [3]]


def test_getitem_tuple(array_list):
    ra = RaggedArray(array_list)
    assert ra[2, 1] == 2


def test_getitem_int(array_list):
    ra = RaggedArray(array_list)
    assert np.all(ra[1] == [2, 1])
    assert np.all(ra == RaggedArray(array_list))


def test_getitem_slice(array_list):
    ra = RaggedArray(array_list)
    subset = ra[1:3]
    true = RaggedArray(array_list[1:3])
    assert subset.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_getitem_list(array_list):
    ra = RaggedArray(array_list)
    subset = ra[[0, 2]]
    true = RaggedArray([array_list[0], array_list[2]])
    assert subset.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_getitem_boolean(array_list):
    ra = RaggedArray(array_list)
    subset = ra[np.array([True, False, False, True])]
    true = RaggedArray([array_list[0], array_list[3]])
    assert subset.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_getitem_empty_boolean(array_list):
    ra = RaggedArray(array_list)
    subset = ra[np.array([False, False, False, False])]
    true = RaggedArray([])
    assert subset.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_getitem_empty_list(array_list):
    ra = RaggedArray(array_list)
    subset = ra[[]]
    true = RaggedArray([])
    assert subset.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_getitem_row_colslice(array_list):
    ra = RaggedArray(array_list)
    subset = ra[[0, 2], :-1]
    true = RaggedArray([row[:-1] for row in [array_list[0], array_list[2]]])
    assert subset.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_getitem_colslice(array_list):
    ra = RaggedArray(array_list)
    subset = ra[:, :-1]
    true = RaggedArray([row[:-1] for row in array_list])
    assert subset.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_add_scalar(array_list):
    ra = RaggedArray(array_list)
    result = np.add(ra, 1)
    true = RaggedArray([[e + 1 for e in row] for row in array_list])
    print(ra, true)
    assert result.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_add_array(array_list):
    ra = RaggedArray(array_list)
    adds = np.arange(4)
    result = np.add(ra, adds[:, None])
    true = RaggedArray([[e + i for e in row] for i, row in enumerate(array_list)])

    assert result.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_add_ra(array_list):
    ra = RaggedArray(array_list)
    adds = np.arange(4)
    result = np.add(ra, ra)
    true = RaggedArray([[e * 2 for e in row] for row in array_list])
    assert result.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_add_operator(array_list):
    ra = RaggedArray(array_list)
    adds = np.arange(4)
    result = ra + ra
    true = RaggedArray([[e * 2 for e in row] for row in array_list])
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


@pytest.mark.skip
def test_rowmean(array_list):
    ra = RaggedArray(array_list)
    m = ra.mean(axis=-1)
    true = np.array([sum(row) / len(row) for row in array_list])
    assert np.all(m == true)
    assert np.all(ra == RaggedArray(array_list))


@pytest.mark.skip
def test_rowstd(array_list):
    ra = RaggedArray(array_list)
    m = ra.std(axis=-1)
    true = np.array([np.std(row) for row in array_list])
    assert np.allclose(m, true)
    assert np.all(ra == RaggedArray(array_list))


def _test_rowmax(array_list):
    ra = RaggedArray(array_list)
    m = ra.max(axis=-1)
    true = np.array([np.max(row) for row in array_list])
    assert np.testing.assert_allclose(m, true)
    assert np.all(ra == RaggedArray(array_list))


def _test_rowmin(array_list):
    ra = RaggedArray(array_list)
    m = ra.min(axis=-1)
    true = np.array([np.min(row) for row in array_list])
    assert np.allclose(m, true)
    assert np.all(ra == RaggedArray(array_list))


def _test_rowargmax(array_list):
    ra = RaggedArray(array_list)
    m = ra.argmax(axis=-1)
    true = np.array([np.argmax(row) for row in array_list])
    assert np.allclose(m, true)
    assert np.all(ra == RaggedArray(array_list))


def _test_rowargmin(array_list):
    ra = RaggedArray(array_list)
    m = ra.argmin(axis=-1)
    true = np.array([np.argmin(row) for row in array_list])
    assert np.allclose(m, true)
    assert np.all(ra == RaggedArray(array_list))


def test_concatenate(array_list):
    ra = RaggedArray(array_list)
    cat = np.concatenate([ra, ra])
    true = RaggedArray(array_list + array_list)
    assert cat.equals(true)
    assert np.all(ra == RaggedArray(array_list))


def test_nonzero(array_list):
    ra = RaggedArray(array_list)
    rows, indices = ra.nonzero()
    assert np.all(rows == [0, 0, 1, 1, 2, 2, 2, 2, 3])
    assert np.all(indices == [1, 2, 0, 1, 0, 1, 2, 3, 0])
    assert np.all(ra == RaggedArray(array_list))


def test_zeros_like(array_list):
    ra = RaggedArray(array_list)
    new = np.zeros_like(ra)
    assert np.all(new._data == 0)
    assert new.shape == ra.shape
    assert np.all(ra == RaggedArray(array_list))


def test_setitem_int(array_list):
    ra = RaggedArray(array_list)
    ra[1] = 10
    array_list[1] = [10, 10]
    assert np.all(ra == RaggedArray(array_list))
    # assert np.all(ra[1] == [2, 1])


def test_setitem_slice(array_list):
    ra = RaggedArray(array_list)
    ra[1:3] = [[10], [20]]
    array_list[1] = [10, 10]
    array_list[2] = [20, 20, 20, 20]
    assert np.all(ra == RaggedArray(array_list))


def test_setitem_list(array_list):
    ra = RaggedArray(array_list)
    ra[[0, 2]] = RaggedArray([[10, 10, 10], [20, 20, 20, 20]])
    array_list[0] = [10, 10, 10]
    array_list[2] = [20, 20, 20, 20]
    assert np.all(ra == RaggedArray(array_list))


def test_setitem_boolean(array_list):
    ra = RaggedArray(array_list)
    ra[np.array([True, False, False, True])] = 0
    array_list[0] = [0, 0, 0]
    array_list[3] = [0]
    assert np.all(ra == RaggedArray(array_list))


def test_rowall(array_list):
    ra = RaggedArray(array_list)
    ba = ra > 0
    s = ba.all(axis=-1)
    true = np.array([all(row) for row in ba])
    assert np.all(s == true)
    assert np.all(ra == RaggedArray(array_list))


def test_rowany(array_list):
    ra = RaggedArray(array_list)
    ba = ra > 2
    s = ba.any(axis=-1)
    true = np.array([any(row) for row in ba])
    assert np.all(s == true)
    assert np.all(ra == RaggedArray(array_list))


def test_reduce(array_list):
    ra = RaggedArray(array_list)
    s = np.add.reduce(ra, axis=-1)
    true = np.array([np.sum(row) for row in array_list])
    assert np.all(s == true)


@pytest.mark.parametrize("op", [np.add, np.subtract, np.bitwise_xor])
def test_accumulate(array_list, op):
    ra = RaggedArray(array_list)
    s = op.accumulate(ra, axis=-1)
    true = RaggedArray([op.accumulate(row) for row in array_list])
    assert np.all(s == true)


def test_cumsum(array_list):
    ra = RaggedArray(array_list)
    s = ra.cumsum(axis=-1)
    true = RaggedArray([np.cumsum(row) for row in array_list])
    assert s.equals(true)


@pytest.mark.skip
def test_cumprod(array_list):
    ra = RaggedArray(array_list)
    s = ra.cumprod(axis=-1)
    true = RaggedArray([np.cumprod(row) for row in array_list])
    assert s.equals(true)


def test_sort(array_list):
    ra = RaggedArray(array_list)
    s = ra.sort(axis=-1)
    true = RaggedArray([np.sort(row) for row in array_list])
    assert s.equals(true)


def test_sum_empty():
    ra = RaggedArray([[1, 2], [], [3]])
    s = ra.sum(axis=-1)
    assert np.all(s == [3, 0, 3])
