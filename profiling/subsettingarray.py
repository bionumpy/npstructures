import time
import numpy as np
from npstructures import RaggedArray
np.random.seed(1000)
n_rows = 80000000
n_subsets = 50000000
row_lens = np.random.randint(1, 3, size=n_rows)
data = np.random.randint(0, 4, size=row_lens.sum(), dtype=np.uint8)
indexes = np.flatnonzero(np.random.randint(0, 2, size=n_rows))
ra = RaggedArray(data, row_lens)
t = time.perf_counter()
last_t = t
for _ in range(10):
    new_data = ra[indexes]
    print(time.perf_counter()-t, time.perf_counter()-last_t, new_data[-10:])
    last_t = time.perf_counter()
