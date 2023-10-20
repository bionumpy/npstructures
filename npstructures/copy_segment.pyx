# cython: infer_types=True
# import numpy as np
cimport cython

ctypedef fused my_type:
    int
    long long
    char
    unsigned char
    short
    unsigned short

ctypedef fused const_type:
    const int
    const long long
    const char
    const unsigned char
    const short
    const unsigned short


@cython.boundscheck(False)
@cython.wraparound(False)
def compute(const_type[:] input_array, my_type[:] output_array, Py_ssize_t[:] starts, Py_ssize_t[:] ends, Py_ssize_t step):
    cdef Py_ssize_t segment = starts.shape[0]
    cdef Py_ssize_t input_i, output_i, slice
    assert tuple(starts.shape) == tuple(ends.shape)

    cdef my_type tmp
    cdef Py_ssize_t x, y
    output_i = 0
    for segment in range(segment):
        for input_i in range(starts[segment], ends[segment], step):
            output_array[output_i] = input_array[input_i]
            output_i += 1
