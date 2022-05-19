from npstructures.bitarray import BitArray
import numpy as np


def test_bitarray():
    values = np.arange(16)
    bit_array = BitArray.pack(values, 4)
    new_values = bit_array.unpack()
    assert np.all(new_values == values)


def test_indexing():
    values = np.arange(16)
    bit_array = BitArray.pack(values, 4)
    for i in range(16):
        assert bit_array[i] == values[i]
