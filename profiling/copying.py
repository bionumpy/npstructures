import numpy as np

from npstructures.raggedshape import native_extract_segments, c_extract_segments, RaggedView2
from time import time
data_size = 100000000
data = np.empty(data_size, dtype=np.uint8)
segment_size = 100
n_segments = data_size//(segment_size*2)
starts = np.arange(n_segments)*(segment_size*2)
lens = np.full(n_segments, segment_size)
view = RaggedView2(starts, lens, 1)
to_shape = view.get_shape()
print('-----------------------')
t = time()
native_extract_segments(data, view, to_shape, 1)
print('native_extract_segments', time()-t)
t = time()
c_extract_segments(data, view, to_shape, 1)
print('c_extract_segments', time()-t)
print('-----------------------')



