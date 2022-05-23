from npstructures.bitarray import BitArray
import numpy as np
from numpy.testing import assert_equal


def test_bitarray():
    values = np.arange(16, dtype=np.uint8)
    bit_array = BitArray.pack(values, 4)
    new_values = bit_array.unpack()
    assert_equal(new_values, values)


def test_bitarray2():
    values = (np.arange(100, dtype=np.uint8)*5 % 7) % 4
    bit_array = BitArray.pack(values, 2)
    new_values = bit_array.unpack()
    assert_equal(new_values, values)


def test_indexing():
    values = np.arange(16, dtype=np.uint8)
    bit_array = BitArray.pack(values, 4)
    for i in range(16):
        assert bit_array[i] == values[i]


def test_list_indexing():
    values = np.arange(16, dtype=np.uint8)
    bit_array = BitArray.pack(values, 4)
    indices = [2, 3, 5, 7]
    for i in range(16):
        assert np.all(bit_array[indices].unpack() == values[indices])


def test_sliding_window():
    values = (np.arange(32, dtype=np.uint8)*5 % 7) % 4
    bit_array = BitArray.pack(values, 2)
    window_size = 27
    windows = bit_array.sliding_window(window_size)
    true = np.convolve(values, 4**np.arange(window_size)[::-1], mode="valid")
    # print([np.binary_repr(n) for n in true])
    np.testing.assert_equal(windows, true)
