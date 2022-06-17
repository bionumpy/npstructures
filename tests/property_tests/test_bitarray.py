from npstructures.bitarray import BitArray
from numpy.testing import assert_equal
import hypothesis.strategies as st
from hypothesis import given, example
import hypothesis.extra.numpy as stnp
import pytest


@pytest.mark.parametrize("n_bits", [1, 2, 4, 8, 16])
@given(array=stnp.arrays(shape=stnp.array_shapes(min_dims=1, max_dims=1, min_side=1), dtype=stnp.integer_dtypes()))
def test_pack_unpack(array, n_bits):
    array = array % 2**n_bits
    bitarray = BitArray.pack(array, n_bits)
    assert_equal(bitarray.unpack(), array)
