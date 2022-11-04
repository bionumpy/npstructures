import numpy as np
from numbers import Number


class BitArray:
    _dtype = np.uint64
    _register_size = _dtype(64)
    _bit_reduce = np.bitwise_or.reduce

    def __init__(self, data: np.ndarray, bit_stride: int, shape: tuple, offset: int = 0):
        self._data = data
        self._bit_stride = self._dtype(bit_stride)
        self._mask = self._dtype(2**bit_stride-1)
        self._shape = shape
        self._offset = self._dtype(offset)
        self._n_entries_per_register = self._register_size//self._bit_stride
        self._shifts = self._dtype(bit_stride)*np.arange(self._n_entries_per_register, dtype=self._dtype)

    @classmethod
    def pack(cls, array: np.ndarray, bit_stride: int) -> "BitArray":
        """Pack a numpy array into n-bit encoded array

        Pack an array where all values can be represented in
        `bit_stride` bits into a `bit_stride`-bit encoded array

        Parameters
        ----------
        cls :
        array : np.ndarray
            Values to be packed
        bit_stride : int
            N-bits per entry

        Returns
        -------
        "BitArray"
            BitArray with packed values

        """
        assert cls._register_size % bit_stride == 0
        n_entries_per_register = cls._register_size // cls._dtype(bit_stride)
        n_registers = cls._dtype((array.size-1)//n_entries_per_register+1)

        data = np.lib.stride_tricks.as_strided(array, shape=(n_registers, n_entries_per_register))
        shifts = cls._dtype(bit_stride)*np.arange(n_entries_per_register, dtype=cls._dtype)
        bits = data[..., 0].astype(cls._dtype)
        for i, shift in enumerate(shifts[1:], 1):
            bits |= (data[..., i].astype(cls._dtype) << cls._dtype(shift))
        return cls(bits, bit_stride, array.shape)

    def unpack(self) -> np.ndarray:
        """Unpack n-bit encoded data into normal numpy array

        Returns
        -------
        np.ndarray
            Unpacked data

        """
        values = ((self._data[:, None] >> self._shifts) & self._mask).ravel()

        return values[:self._shape[0]]

    def __getitem__(self, idx: list):
        if isinstance(idx, list):
            idx = np.asanyarray(idx)
        register_idx = self._dtype(idx+self._offset) // (self._n_entries_per_register)
        register_offset = self._dtype(idx + self._offset) % (self._n_entries_per_register)
        if isinstance(idx, Number):
            return (self._data[register_idx] >> (register_offset*self._bit_stride)) & self._mask
        if isinstance(idx, np.ndarray):
            array = self._data[register_idx] >> (register_offset*self._bit_stride) & self._mask
            return self.pack(array, self._bit_stride)

    def sliding_window(self, window_size: int) -> np.ndarray:
        """Return sliding windows of bit-packed values

        For window sizes less than `n_entries_per_register`. Return
        one register per window

        Parameters
        ----------
        window_size : int
            how many entries in each window

        Returns
        -------
        np.ndarray
            self.size-window_size+1 windows


        """
        mask = (~self._dtype(0)) >> self._dtype(self._register_size-window_size * self._bit_stride)
        rev_shifts = self._shifts[::-1] + self._bit_stride
        res = self._data[:, None] >> self._shifts
        res[:-1] |= self._data[1:, None] << rev_shifts
        res &= mask
        return res.ravel()[:self._shape[0]-window_size+1]
