import pytest
from tests.npbackend import np
from npstructures import RaggedArray
from collections import defaultdict
from numpy.testing import assert_equal


@pytest.fixture
def array_list():
    return [[0, 1, 2], [2, 1], [1, 2, 3, 4], [3]]


def assert_ra_equal(a, b):
    np.testing.assert_equal(a.ravel(), b.ravel())
    assert a.shape[0] == b.shape[0]
    assert np.all(a.shape[1] == b.shape[1])


def test_shape(array_list):
    ra = RaggedArray(array_list)
    assert ra.shape[0] == 4
    assert np.all(ra.shape[1] == np.array([3, 2, 4, 1]))


def test_lenghts(array_list):
    ra = RaggedArray(array_list)
    assert np.all(ra.lengths == np.array([3, 2, 4, 1]))


def test_two_indexing(array_list):
    ra = RaggedArray(array_list)
    a = ra[1:]
    b = a[:2]
    assert_ra_equal(b, RaggedArray(array_list[1:3]))


def test_two_indexing_row_col(array_list):
    ra = RaggedArray(array_list)
    a = ra[1:, 1:]
    b = a[:2, :-1]
    assert_ra_equal(b, RaggedArray([r[1:-1] for r in array_list[1:3]]))


def test_two_indexing_row_col2(array_list):
    ra = RaggedArray(array_list)
    a = ra[1:]
    b = a[:2, :-1]
    assert_ra_equal(b, RaggedArray([r[:-1] for r in array_list[1:3]]))


def test_two_indexing_row_col3(array_list):
    ra = RaggedArray(array_list)
    a = ra[1:, 1:]
    b = a[:2]
    assert_ra_equal(b, RaggedArray([r[1:] for r in array_list[1:3]]))


def test_two_indexing_row_n(array_list):
    ra = RaggedArray(array_list)
    a = ra[1:, 1:]
    b = a[0]
    assert_equal(b, array_list[1][1:])


@pytest.mark.cupy
def test_getitem_tuple(array_list):
    ra = RaggedArray(array_list)
    assert ra[2, 1] == 2


@pytest.mark.cupy
def test_getitem_int(array_list):
    ra = RaggedArray(array_list)
    assert np.all(ra[1] == np.asanyarray([2, 1]))
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_getitem_slice(array_list):
    ra = RaggedArray(array_list)
    subset = ra[1:3]
    true = RaggedArray(array_list[1:3])
    assert_ra_equal(subset, true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_getitem_list(array_list):
    ra = RaggedArray(array_list)
    subset = ra[[0, 2]]
    true = RaggedArray([array_list[0], array_list[2]])
    assert subset.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_getitem_boolean(array_list):
    ra = RaggedArray(array_list)
    subset = ra[np.array([True, False, False, True])]
    true = RaggedArray([array_list[0], array_list[3]])
    assert subset.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_getitem_empty_boolean(array_list):
    ra = RaggedArray(array_list)
    subset = ra[np.array([False, False, False, False])]
    true = RaggedArray([])
    assert subset.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_getitem_empty_list(array_list):
    ra = RaggedArray(array_list)
    subset = ra[[]]
    true = RaggedArray([])
    assert subset.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_getitem_row_colslice(array_list):
    ra = RaggedArray(array_list)
    subset = ra[[0, 2], :-1]
    true = RaggedArray([row[:-1] for row in [array_list[0], array_list[2]]])
    assert subset.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_getitem_colslice(array_list):
    ra = RaggedArray(array_list)
    subset = ra[:, :-1]
    true = RaggedArray([row[:-1] for row in array_list])
    assert subset.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


# @pytest.mark.cupy # Operator works
def test_add_scalar(array_list):
    ra = RaggedArray(array_list)
    result = np.add(ra, 1)
    #result = ra + 1
    true = RaggedArray([[e + 1 for e in row] for row in array_list])
    print(ra, true)
    assert result.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


# @pytest.mark.cupy # Operator does not work
def test_add_array(array_list):
    ra = RaggedArray(array_list)
    adds = np.arange(4)
    result = np.add(ra, adds[:, None])
    #result = ra + adds[:, None]
    true = RaggedArray([[e + i for e in row] for i, row in enumerate(array_list)])

    assert result.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


#@pytest.mark.cupy 
def test_add_ra(array_list):
    ra = RaggedArray(array_list)
    adds = np.arange(4)
    result = np.add(ra, ra)
    true = RaggedArray([[e * 2 for e in row] for row in array_list])
    assert result.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_add_operator(array_list):
    ra = RaggedArray(array_list)
    adds = np.arange(4)
    result = ra + ra
    true = RaggedArray([[e * 2 for e in row] for row in array_list])
    assert result.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_sum(array_list):
    ra = RaggedArray(array_list)
    assert ra.sum() == sum(e for row in array_list for e in row)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_rowsum(array_list):
    ra = RaggedArray(array_list)
    s = ra.sum(axis=-1)
    true = np.array([sum(row) for row in array_list])
    assert np.all(s == true)
    assert_ra_equal(ra, RaggedArray(array_list))


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


#@pytest.mark.cupy
def test_concatenate(array_list):
    ra = RaggedArray(array_list)
    cat = np.concatenate([ra, ra])
    true = RaggedArray(array_list + array_list)
    assert cat.equals(true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_nonzero(array_list):
    ra = RaggedArray(array_list)
    rows, indices = ra.nonzero()
    assert np.all(rows == np.array([0, 0, 1, 1, 2, 2, 2, 2, 3]))
    assert np.all(indices == np.array([1, 2, 0, 1, 0, 1, 2, 3, 0]))
    assert_ra_equal(ra, RaggedArray(array_list))

#@pytest.mark.cupy
def test_zeros_like(array_list):
    ra = RaggedArray(array_list)
    new = np.zeros_like(ra)
    assert np.all(new.ravel() == 0)
    assert new._shape == ra._shape
    assert np.all(ra == RaggedArray(array_list))


@pytest.mark.cupy
def test_setitem_int(array_list):
    ra = RaggedArray(array_list)
    ra[1] = 10
    array_list[1] = [10, 10]
    assert_ra_equal(ra, RaggedArray(array_list))
    # assert np.all(ra[1] == [2, 1])


#@pytest.mark.cupy
def test_setitem_slice(array_list):
    ra = RaggedArray(array_list)
    ra[1:3] = [[10], [20]]
    array_list[1] = [10, 10]
    array_list[2] = [20, 20, 20, 20]
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_setitem_list(array_list):
    ra = RaggedArray(array_list)
    ra[[0, 2]] = RaggedArray([[10, 10, 10], [20, 20, 20, 20]])
    array_list[0] = [10, 10, 10]
    array_list[2] = [20, 20, 20, 20]
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_setitem_boolean(array_list):
    ra = RaggedArray(array_list)
    ra[np.array([True, False, False, True])] = 0
    array_list[0] = [0, 0, 0]
    array_list[3] = [0]
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_setitem_ragged_boolean(array_list):
    ra = RaggedArray(array_list)
    flat_mask = np.tile([True, False], ra.size//2+1)[:ra.size]
    mask = RaggedArray(flat_mask, ra._shape)
    ra[mask] = 100
    np.testing.assert_array_equal(ra.ravel()[flat_mask], 100)


def test_setitem_fails(array_list):
    ra = RaggedArray(array_list)
    mask = ra
    with pytest.raises(TypeError):
        ra[mask] = 2


@pytest.mark.cupy
def test_rowall(array_list):
    ra = RaggedArray(array_list)
    ba = ra > 0
    s = ba.all(axis=-1)
    true = np.array([all(row) for row in ba])
    assert np.all(s == true)
    assert_ra_equal(ra, RaggedArray(array_list))


@pytest.mark.cupy
def test_rowany(array_list):
    ra = RaggedArray(array_list)
    ba = ra > 2
    s = ba.any(axis=-1)
    true = np.array([any(row) for row in ba])
    assert np.all(s == true)
    assert_ra_equal(ra, RaggedArray(array_list))


#@pytest.mark.cupy
def test_reduce(array_list):
    ra = RaggedArray(array_list)
    s = np.add.reduce(ra, axis=-1)
    true = np.array([np.sum(row) for row in array_list])
    assert np.all(s == true)


#@pytest.mark.cupy
@pytest.mark.parametrize("op", [np.add, np.subtract, np.bitwise_xor])
def test_accumulate(array_list, op):
    ra = RaggedArray(array_list)
    s = op.accumulate(ra, axis=-1)
    true = RaggedArray([op.accumulate(row) for row in array_list])
    assert np.all(s == true)


@pytest.mark.cupy
def test_cumsum(array_list):
    ra = RaggedArray(array_list)
    s = ra.cumsum(axis=-1)
    true = RaggedArray([np.cumsum(np.asanyarray(row)) for row in array_list])
    assert s.equals(true)


@pytest.mark.skip
def test_cumprod(array_list):
    ra = RaggedArray(array_list)
    s = ra.cumprod(axis=-1)
    true = RaggedArray([np.cumprod(row) for row in array_list])
    assert s.equals(true)


#@pytest.mark.cupy
def test_sort(array_list):
    ra = RaggedArray(array_list)
    s = ra.sort(axis=-1)
    true = RaggedArray([np.sort(row) for row in array_list])
    assert s.equals(true)


@pytest.mark.cupy
def test_sum_empty():
    ra = RaggedArray([[1, 2], [], [3]])
    s = ra.sum(axis=-1)
    assert np.all(s == np.asanyarray([3, 0, 3]))


@pytest.mark.parametrize("func", [np.all, np.any, np.sum, np.prod])
def test_reduction_functions_with_multiple_empty_rows_at_end(func):
    nested_list = [[], [True, True], [True, False], [False, False], [], [], []]
    correct = [func(l) for l in nested_list]
    ra = RaggedArray(nested_list)
    assert np.all(correct == func(ra, axis=-1))


def test_subset_with_boolean_ragged_array():
    ra = RaggedArray([[], [1, 2, 3], [1, 2], [1], [], []])
    subset_with = RaggedArray([[], [True, False, True], [True, False], [True], [], []])
    assert np.all(ra.subset(subset_with) == RaggedArray([[], [1, 3], [1], [1], [], []]))


@pytest.mark.skip('depracated')
def test_as_padded_matrix():
    ra = RaggedArray([[1, 2, 3], [1, 2], [1]])
    padded = ra.as_padded_matrix(side="right")
    assert np.all(padded == [[1, 2, 3], [1, 2, 0], [1, 0, 0]])

    padded = ra.as_padded_matrix(side="left")
    assert np.all(padded == [[1, 2, 3], [0, 1, 2], [0, 0, 1]])

    ra = RaggedArray([[1], [], [1], [1, 2], [1, 2, 3]])
    padded = ra.as_padded_matrix(fill_value=-1, side="left")
    assert np.all(padded == [[-1, -1, 1], [-1, -1, -1], [-1, -1, 1], [-1, 1, 2], [1, 2, 3]])

    padded = ra.as_padded_matrix(fill_value=-1, side="right")
    assert np.all(padded == [[1, -1, -1], [-1, -1, -1], [1, -1, -1], [1, 2, -1], [1, 2, 3]])


@pytest.mark.parametrize("array_list", [
    [[1, 2, 3], [1, 2], [5]],
    [[10], [], [100, 100, 100]],
    [[], [5.3, 4.0], [1, 2, 3, 4], []]
])
def test_sum_and_mean_axis_0(array_list):
    ra = RaggedArray(array_list)
    columns = defaultdict(list)
    for row in array_list:
        for i in range(len(row)):
            columns[i].append(row[i])

    ra_sum = ra.sum(axis=0)
    ra_mean = ra.mean(axis=0)

    for i in range(len(ra)):
        assert ra_sum[i] == sum(columns[i])
        assert ra_mean[i] == np.mean(columns[i])


def test_sum_on_boolean_array():
    ra = RaggedArray([[True, True, False], [True, False]])
    s = np.sum(ra, axis=0)
    print(s.dtype)
    assert np.issubdtype(s.dtype, np.integer)
    assert np.array_equal(s, [2, 1, 0])


def test_repr(array_list):
    ra = RaggedArray(array_list)
    r = repr(ra)
    for row in array_list:
        for c in row:
            assert str(c) in r
