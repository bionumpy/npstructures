import pytest
import numpy as np
from npstructures import HashTable, Counter

@pytest.fixture
def array_list():
    return [[0, 1, 2],
            [2, 1],
            [1, 2, 3, 4],
            [3]]

@pytest.mark.parametrize("cls", [HashTable])
def test_lookup(cls):
    keys = [0, 3, 7, 11, 13, 17, 19, 23, 29, 31]
    values = np.arange(len(keys))
    table = cls(keys, values, 7)
    assert np.all(table[keys] == values)
    assert np.all(table[keys][::-1] == values[::-1])

@pytest.mark.parametrize("cls", [HashTable])#, IHashTable])
def test_lookup_small(cls):
    keys = [0, 3]
    values = np.arange(len(keys))
    table = cls(keys, values, 17)
    assert np.all(table[keys] == values)
    assert np.all(table[keys][::-1] == values[::-1])

def test_count():
    keys = [0, 3, 7, 11, 13, 17, 19, 23, 29, 31]
    counter = Counter(keys, mod=17)
    samples = [9,0, 3, 12, 3, 7,10,7,7, 2, 3, 0]
    counter.count(samples)
    assert np.all(counter[[0,3,7]] == [2, 3, 3])
    assert np.all(counter[[11, 13, 17, 19, 23, 29, 31]] == 0)
