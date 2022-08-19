import cupy as cp

from ..raggedshape import RaggedShape, ViewBase

class CPRaggedShape(RaggedShape):
    def __init__(self, codes, is_coded=False):
        codes = cp.asanyarray(codes, dtype=self._dtype)
        super().__init__(codes, is_coded=is_coded)

    """def _get_accumulation_func(self, dtype):
        if dtype == bool:
            return NotImplemented
        return np.cumsum"""
