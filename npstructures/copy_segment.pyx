# cython: infer_types=True
# import numpy as np
cimport cython
cimport numpy as np
from numpy import int8

ctypedef fused my_type:
    int
    double
    long long
    np.uint8_t
    np.int8_t
    np.uint16_t
    np.int16_t




@cython.boundscheck(False)
@cython.wraparound(False)
def compute(my_type[:] input_array, my_type[:] output_array, Py_ssize_t[:] starts, Py_ssize_t[:] ends):
    cdef Py_ssize_t segment = starts.shape[0]
    cdef Py_ssize_t input_i, output_i, slice
    assert tuple(starts.shape) == tuple(ends.shape)

    cdef my_type tmp
    cdef Py_ssize_t x, y
    output_i = 0
    for segment in range(segment):
        for input_i in range(starts[segment], ends[segment]):
            output_array[output_i] = input_array[input_i]
            output_i += 1
