from npstructures.hashtable import HashTable, IHashTable
import timeit
import numpy as np
np.random.seed(10000)
def get_random_data(n_keys, key_space, n_samples):
    keys = np.cumsum(np.random.randint(1, key_space, n_keys))
    samples = np.random.choice(keys, size=n_samples)
    return keys, samples


keys, samples = get_random_data(10000000, 7, 1000000)
table = HashTable(keys, np.arange(keys.size), keys.size*3-1)

for cls in (IHashTable, HashTable):
    print(cls.__name__)
    table = cls(keys, np.arange(keys.size), keys.size*3-1)
    table[samples]
    print("running")
    print(timeit.repeat("table[samples]", globals=globals(), number=1, repeat=1))
