import numpy as np
import cupy as cp

import npstructures

rs = npstructures.RaggedShape(codes=[2, 3, 4])
print(type(rs))
print(rs)
print()

ra = npstructures.RaggedArray(data=np.arange(100), shape=[20, 20, 50, 10])
print(type(ra))
print(ra)
print()

ra = ra + 1
print(type(ra))
print(ra)
print()

npstructures.set_backend(cp)

rs = npstructures.RaggedShape(codes=[2, 3, 4])
print(type(rs))
print(rs)
print()

ra = npstructures.RaggedArray(data=np.arange(100), shape=[20, 20, 50, 10])
print(type(ra))
print(ra)
print()

ra = ra + 1
print(type(ra))
print(ra)
