from npstructures.hashtable import HashTable, IHashTable, Counter
import timeit
import cProfile
import pstats
import numpy as np
np.random.seed(10000)
def get_random_data(n_keys, key_space, n_samples):
    keys = np.cumsum(np.random.randint(1, key_space, n_keys))
    samples = np.random.choice(keys, size=n_samples)
    return keys, samples


keys, samples = get_random_data(80000000, 7, 5000000)
# table = HashTable(keys, np.arange(keys.size), keys.size*3-1)
counter = Counter(keys, keys.size*3-1)
p_stats_name = "profiling/.hash_table.txt"
cProfile.run("counter.count(samples)", p_stats_name)
stats = pstats.Stats(p_stats_name)
stats.sort_stats("cumulative")
stats.print_stats()


for cls in (HashTable,):
    print(cls.__name__)
    table = cls(keys, np.arange(keys.size), keys.size*3-1)
    table[samples]
    print("running")
    print(timeit.repeat("table[samples]", globals=globals(), number=1, repeat=1))
