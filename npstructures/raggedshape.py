from numbers import Number
import numpy as np
import traceback

HANDLED_FUNCTIONS = {}

def implements(np_function):
   "Register an __array_function__ implementation for DiagonalArray objects."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator


class RaggedShape:
    def __init__(self, offsets):
        self._offsets = np.asanyarray(offsets)

    def ravel_multi_index(self, indices):
        return self._offsets[indices[0]]+indices[1]

    def unravel_multi_index(self, flat_indices):
        rows = np.searchsorted(self._offsets, flat_indices, side="right")-1
        cols = flat_indices-self._offsets[rows]
        return rows, cols

    def index_array(self):
        diffs = np.zeros(self._offsets[-1], dtype=int)
        diffs[self._offsets[1:-1]] = 1
        return np.cumsum(diffs)

    @property
    def lengths(self):
        return self.ends-self.starts

    @property
    def starts(self):
        return self._offsets[:-1]

    @property
    def ends(self):
        return self._offsets[1:]

    @property
    def size(self):
        return self._offsets[-1]

    @classmethod
    def asanyshape(cls, shape):
        if isinstance(shape, RaggedShape):
            return shape
        return cls(shape)

    def __getitem__(self, index):
        if isinstance(index, slice):
            new_offsets = self._offsets[index.start:index.stop+1]-self._offsets[index.start]
            return RaggedShape(new_offsets)
        return NotImplemented


    def view(self, indices):
        return RaggedView(self.starts[indices], self.ends[indices])

    @property
    def n_rows(self):
        return self.starts.size

    def to_dict(self):
        return {"offsets": self._offsets}

    @staticmethod
    def from_dict(d):
        if "offsets" in d:
            return CodedRaggedShape(d["offsets"])
        else:
            return CodedRaggedShape(d["codes"], d["row_bits"], d["index_bits"])
        
    def __eq__(self, other):
        return np.all(self.starts == other.starts) and np.all(self.lengths == other.lengths)

    def __str__(self):
        return f"{self.__class__.__name__}({self.starts}, {self.lengths})"
    __repr__= __str__



class RaggedView:
    def __init__(self, starts, ends):
        self.starts = starts
        self.ends = ends

    @property
    def lengths(self):
        return self.ends-self.starts

class CodedRaggedShape(RaggedShape):
    def __init__(self, offsets):
        if isinstance(offsets, np.ndarray) and offsets.dtype==np.uint64:
            self._codes = offsets
        else:
            offsets = np.asanyarray(offsets, dtype=np.int32)
            starts = offsets[:-1]
            lens = np.diff(offsets)
            assert lens.dtype==np.int32, lens.dtype
            self._codes = np.hstack((starts[:, None], lens[:, None])).flatten().view(np.uint64)

    @property
    def lengths(self):
        if isinstance(self._codes, Number):
            return np.atleast_1d(self._codes).view(np.int32)[1]
        return self._codes.view(np.int32)[1::2]

    @property
    def starts(self):
        if isinstance(self._codes, Number):
            return np.atleast_1d(self._codes).view(np.int32)[0]

        return self._codes.view(np.int32)[::2]

    @property
    def ends(self):
        return self.starts+self.lengths
        # return ((self._codes >> self._row_bits) & np.uint64(2**self._index_bits-1)).view(int)

    @property
    def size(self):
        return np.sum(self._codes.view(np.int32)[-2:])
    
    def ravel_multi_index(self, indices):
        return (self.starts[indices[0]]+indices[1]).view(np.int32)

    def unravel_multi_index(self, flat_indices):
        starts = self.starts
        rows = np.searchsorted(starts, flat_indices, side="right")-1
        cols = flat_indices-starts[rows]
        return rows, cols #TODO: Cast rows to int32

    def index_array(self):
        diffs = np.zeros(self.size, dtype=np.int32)
        diffs[self.starts[1:]] = 1
        return np.cumsum(diffs)

    def view(self, indices):
        return CodedRaggedView(self._codes[indices])

    def __getitem__(self, index):
        if not isinstance(index, slice) or isinstance(index, Number):
            return NotImplemented
        new_codes = self._codes[index].view(np.int32)
        new_codes[::2] -= new_codes[0]
        return self.__class__(new_codes.view(np.uint64))

    def to_dict(self):
        return {"codes": self._codes}


class _CodedRaggedShape(RaggedShape):
    def to_dict(self):
        return {"codes": self._codes, "row_bits": self._row_bits, "index_bits": self._index_bits}

    def _get_bitshifts(self, offsets):
        max_row = np.max(offsets[1:]-offsets[:-1])
        n_row_len_bits = np.uint64(np.log2(max_row)+1)
        n_index_bits = np.uint64(np.log2(offsets[-1])+1)
        assert n_index_bits*2+n_row_len_bits<=64
        return n_row_len_bits, n_index_bits
        
    def __init__(self, offsets, row_bits=None, index_bits=None):
        if row_bits is None:
            offsets = np.asanyarray(offsets, dtype=np.uint64)
            self._row_bits, self._index_bits = self._get_bitshifts(offsets)
            self._codes = offsets[:-1] << (self._row_bits+self._index_bits)
            row_lens = offsets[1:]-offsets[:-1]
            self._codes+= (offsets[:-1]+row_lens)<<self._row_bits
            self._codes+= row_lens
        else:
            self._codes = offsets
            self._row_bits = row_bits
            self._index_bits = index_bits
        assert self._codes.dtype==np.uint64
        assert self._row_bits.dtype==np.uint64
        assert self._index_bits.dtype==np.uint64


    def __eq__(self, other):
        #assert self.__class__==other.__class__, (self.__class__, other.__class__)
        return np.all(self.starts == other.starts) and np.all(self.lengths == other.lengths)

    def __getitem__(self, index):
        new_codes = self._codes[index]
        init = new_codes[0] & ~np.uint64(2**(self._row_bits+self._index_bits)-1)
        init += init >> self._index_bits
        return self.__class__(new_codes-init, self._row_bits, self._index_bits)

    def __str__(self):
        return f"CRS({self.starts}, {self.ends}, {self.lengths})"
    __repr__= __str__

    @property
    def lengths(self):
        return (self._codes & np.uint64(2**self._row_bits-1)).view(int)

    @property
    def starts(self):
        # print("---------------------")
        # traceback.print_stack()
        return (self._codes >> (self._row_bits+self._index_bits)).view(int)

    @property
    def ends(self):
        return ((self._codes >> self._row_bits) & np.uint64(2**self._index_bits-1)).view(int)

    @property
    def size(self):
        bitmask = np.uint64(2**self._index_bits-1)
        ret = (self._codes[-1] >> self._row_bits) & np.uint64(2**self._index_bits-1)
        return ret.view(int)
    
    def ravel_multi_index(self, indices):
        return (self.starts[indices[0]]+indices[1]).view(int)

    def unravel_multi_index(self, flat_indices):
        starts = self.starts
        rows = np.searchsorted(starts, flat_indices, side="right")-1
        cols = flat_indices-starts[rows]
        return rows, cols

    def index_array(self):
        diffs = np.zeros(self.size, dtype=int)
        diffs[self.starts[1:]] = 1
        return np.cumsum(diffs)

    def view(self, indices):
        return CodedRaggedView(self._codes[indices], self._row_bits, self._index_bits)

class _CodedRaggedView(CodedRaggedShape, RaggedView):
    def __init__(self, codes, row_bits, index_bits):
        self._codes = codes
        self._row_bits = row_bits
        self._index_bits = index_bits

    def __getitem__(self, index):
        return self.__class__(self._codes[index], self._row_bits, self._index_bits)

class CodedRaggedView(CodedRaggedShape, RaggedView):
    def __init__(self, codes):
        self._codes = codes

    def __getitem__(self, index):
        return self.__class__(self._codes[index])
