from ..bitarray import BitArray, np


class CpBitArray(BitArray):
    _bit_reduce = np.sum
