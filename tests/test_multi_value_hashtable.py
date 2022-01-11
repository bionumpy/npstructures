from npstructures.multi_value_hashtable import MultiValueHashTable
import numpy as np


def test():
    h = MultiValueHashTable.from_keys_and_values([1, 2, 3, 1], {"nodes": np.array([1, 2, 3, 10]), "offsets": np.array([5, 3, 2, 100])}, mod=11)
    assert np.all(h[1]["nodes"] == [1, 10])
    assert np.all(h[2]["offsets"] == [3])

test()