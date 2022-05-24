import pytest
from npstructures import Counter
import numpy as np

np.random.seed(1)

def test_with_big_keys():
    keys = np.unique(np.random.randint(0, 4 ** 30, 1000, dtype=np.int64))
    values_to_be_counted = np.concatenate([keys, keys])  # keys twice
    c = Counter(keys)
    c.count(values_to_be_counted)
    counts = c[keys]
    assert np.all(counts == 2)


def test_return_counts():
    counter = Counter([1, 5, 10, 15])
    counts = counter.count([1, 1, 10, 15, 5, 5], return_counts=True)
    assert np.all(counter[[1, 5, 10, 15]] == [0, 0, 0, 0])
    assert np.all(counts[[1, 5, 10, 15]] == [2, 2, 1, 1])

    counter = Counter([1, 5, 10, 15], [1, 2, 3, 4])
    counts = counter.count([1, 1, 10, 15, 5, 5], return_counts=True)
    assert np.all(counter[[1, 5, 10, 15]] == [1, 2, 3, 4])
    assert np.all(counts[[1, 5, 10, 15]] == [3, 4, 4, 5])
