# cython: infer_types=True
import numpy as np
cimport cython

ctypedef fused my_type:
    int
    double
    long long


@cython.boundscheck(False)
@cython.wraparound(False)
def compute(my_type[:] input_array, my_type[:] output_array, int[:] starts, int[:] lens):
    cdef Py_ssize_t n_slices = starts.shape[0]
    cdef Py_ssize_t input_i, output_i, slice
    assert tuple(starts.shape) == tuple(lens.shape)

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    cdef my_type tmp
    cdef Py_ssize_t x, y
    output_i = 0
    for slice in range(n_slices):
        for input_i in range(starts[slice], starts[slice] + lens[slice]):
            output_array[output_i] = input_array[input_i]

