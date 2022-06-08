from ..raggedarray import RaggedArray

class CPRaggedArray(RaggedArray):
    def _accumulate(*args, **kwargs):
        return NotImplemented

    def _reduce(*args, **kwargs):
        return NotImplemented


    def max(*args, **kwargs):
        return NotImplemented

    def _row_accumulate(self, operator, dtype=None):
        return NotImplemented

