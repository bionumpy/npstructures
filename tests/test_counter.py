from npstructures import Counter
from tests.npbackend import np

np.random.seed(1)


def test_with_big_keys():
    keys = np.unique(np.random.randint(0, 4 ** 30, 1000, dtype=np.int64))
    values_to_be_counted = np.concatenate([keys, keys])  # keys twice
    c = Counter(keys)
    c.count(values_to_be_counted)
    counts = c[keys]
    assert np.all(counts == 2)
