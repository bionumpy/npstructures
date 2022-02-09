from npstructures.arrayfunctions import diff
from npstructures.raggedarray import RaggedArray
import pytest
import numpy as np


@pytest.fixture
def array_list():
    return [[0, 1, 2],
            [2, 1],
            [1, 2, 3, 4],
            [3]]

@pytest.mark.parametrize("n", [1, 2])
def test_diff_1(array_list, n):
    ra = RaggedArray(array_list)
    d = np.diff(ra, n, axis=-1)
    true = RaggedArray([np.diff(row, n) for row in array_list])
    assert d.equals(true)
