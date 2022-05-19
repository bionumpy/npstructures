import numpy as np
from numbers import Number


class BitArray:

    _register_size = 64
    
    def __init__(self, data, bit_stride, shape, offset=0):
        self._data = data
        print(self._data)

        self._bit_stride = bit_stride
        self._mask = 2**bit_stride-1
        self._shape = shape
        self._offset = offset
        self._n_entries_per_register = self._register_size//self._bit_stride

    @classmethod
    def pack(cls, array, bit_stride):
        assert cls._register_size % bit_stride == 0
        n_entries_per_register = cls._register_size // bit_stride
        n_registers = (array.size-1)//n_entries_per_register+1
        data = np.lib.stride_tricks.as_strided(array, shape=(n_registers, n_entries_per_register))
        shifts = bit_stride*np.arange(n_entries_per_register)
        shifted = data << shifts
        bits = np.bitwise_or.reduce(shifted, axis=-1)
        return cls(bits, bit_stride, array.shape)

    def unpack(self):
        n_entries_per_register = self._register_size // self._bit_stride
        shifts = self._bit_stride*np.arange(n_entries_per_register)
        mask = 2**(self._bit_stride)-1
        values = ((self._data[:, None] >> shifts) & mask).ravel()
        return values[:self._shape[0]]

    def __bitshift__(self, n_bits):
        self._data >> n_bits
        result = sequence[:, None] >> self._shifts
        result[:-1] |= sequence[1:, None] << self._rev_shifts

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = np.asanyarray(idx)
        register_idx = (idx+self._offset) // (self._n_entries_per_register)
        register_offset = (idx + self._offset) % (self._n_entries_per_register)
        if isinstance(idx, Number):
            return (self._data[register_idx] >> (register_offset*self._bit_stride)) & self._mask
        if isinstance(idx, np.ndarray):
            array = self._data[register_idx] >> (register_offset*self._bit_stride) & self._mask
            return self.pack(array, self._bit_stride)

    def get_kmers_with_buffer(self, sequence):
        res = sequence[:-1, None] >> self._shifts
        res |= sequence[1:, None] << self._rev_shifts
        return res

    def get_kmers(self, sequence, is_internal=False):
        assert sequence.dtype == self._dtype, sequence.dtype
        result = sequence[:, None] >> self._shifts
        result[:-1] |= sequence[1:, None] << self._rev_shifts
        return result
