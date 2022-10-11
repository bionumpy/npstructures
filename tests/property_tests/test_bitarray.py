from tests.npbackend import np
from npstructures.bitarray import BitArray
from numpy.testing import assert_equal
import hypothesis.strategies as st
from hypothesis import given, example
import hypothesis.extra.numpy as stnp
import pytest


@given(array=stnp.arrays(shape=stnp.array_shapes(min_dims=1, max_dims=1, min_side=1), dtype=stnp.integer_dtypes()),
       n_bits=st.sampled_from([1, 2, 4, 8, 16]))
def test_pack_unpack(array, n_bits):
    array = array % 2**n_bits
    bitarray = BitArray.pack(array, n_bits)
    assert_equal(bitarray.unpack(), array)


@given(array=stnp.arrays(shape=stnp.array_shapes(min_dims=1, max_dims=1, min_side=1), dtype=stnp.integer_dtypes()),
       n_bits=st.sampled_from([1, 2, 4, 8, 16]),
       window_size=st.sampled_from([1, 2, 4, 8, 16]))
@example(array=np.array([0], dtype=np.int8), n_bits=2, window_size=3)
@example(array=np.array([0, 0, 0, 0, 0], dtype=np.int8), n_bits=16, window_size=8)
def test_sliding_window(array, n_bits, window_size):
    window_size = min(window_size, array.size, 64//n_bits)
    array = array % 2**n_bits
    bit_slided = BitArray.pack(array, n_bits).sliding_window(window_size)
    slided = np.lib.stride_tricks.sliding_window_view(array, window_size)
    unpacked = np.array([BitArray(np.atleast_1d(row), n_bits, (window_size, )).unpack()
                         for row in bit_slided])
    assert_equal(unpacked, slided)
