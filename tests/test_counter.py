from npstructures import Counter
import numpy as np
np.random.seed(1)

def test_with_big_keys():
    keys = np.unique(np.random.randint(0, 4**30, 1000, dtype=np.int64))
    values_to_be_counted = np.concatenate([keys, keys])  # keys twice
    c = Counter(keys, encode32=True)
    print("##################")
    c.count(values_to_be_counted)
    print("##################")
    counts = c[keys]
    print("##################")
    assert np.all(counts == 2)
