from npstructures.arrayfunctions import diff
from npstructures.raggedarray import RaggedArray
import pytest
from tests.npbackend import np


@pytest.fixture
def array_list():
    return [[0, 1, 2], [2, 1], [1, 2, 3, 4], [3]]


@pytest.fixture
def array_list2():
    return [[0, 1, 2, 2], [2, 1, 1, 3], [1, 2, 3], [3, 3, 3]]


@pytest.fixture
def array_list3():
    return [[0, 1, 2, 2], [], [1, 2, 3], [3, 3, 3], []]


@pytest.mark.parametrize("n", [1, 2])
def test_diff_1(array_list, n):
    ra = RaggedArray(array_list)
    d = np.diff(ra, n, axis=-1)
    true = RaggedArray([np.diff(row, n) for row in array_list])
    assert d.equals(true)


def _test_convolve(array_list2):
    ra = RaggedArray(array_list2)
    convolved = np.convolve(ra)


def test_unique(array_list2):
    ra = RaggedArray(array_list2)
    unique = np.unique(ra, axis=-1)
    true = RaggedArray([np.unique(row) for row in array_list2])
    assert unique.equals(true)


def test_unique_with_counts(array_list2):
    ra = RaggedArray(array_list2)
    unique, counts = np.unique(ra, axis=-1, return_counts=True)
    tu, tc = (
        RaggedArray(a)
        for a in zip(*(np.unique(row, return_counts=True) for row in array_list2))
    )
    assert unique.equals(tu)
    assert counts.equals(tc)


def test_unique_2(array_list3):
    ra = RaggedArray(array_list3)
    unique = np.unique(ra, axis=-1)
    true = RaggedArray([np.unique(row) for row in array_list3])

    assert unique.equals(true)


def test_unique_bug():
    array_list3 = [[], [0, 0], [], [4], [0, 0]]
    ra = RaggedArray(array_list3)
    unique, counts = np.unique(ra, axis=-1, return_counts=True)
    true = RaggedArray([np.unique(row) for row in array_list3])
    true_counts = RaggedArray(
        [np.unique(row, return_counts=True)[1] for row in array_list3]
    )
    assert unique.equals(true)
    assert counts.equals(true_counts)
