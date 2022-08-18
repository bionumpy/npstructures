import cupy as cp

from ..raggedshape import RaggedShape, ViewBase

class CPRaggedShape(RaggedShape):
    def __init__(self, codes, is_coded=False):
        print("CPRaggedShape constructor")

        codes = cp.asanyarray(codes, dtype=self._dtype)
        super().__init__(codes, is_coded=is_coded)

        """if is_coded:
            super().__init__(codes)
        else:
            lengths = cp.asanyarray(codes, dtype=self._dtype)
            starts = cp.pad(cp.cumsum(lengths, dtype=self._dtype)[:-1], pad_width=1, mode="constant")[:-1]

            super().__init__(starts, lengths)

        self._is_coded = True"""


    """def _get_accumulation_func(self, dtype):
        if dtype == bool:
            return NotImplemented
        return np.cumsum"""
