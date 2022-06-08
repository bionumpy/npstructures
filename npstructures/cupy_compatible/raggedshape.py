
class CPRaggedShape(RaggedArray):
    def _get_accumulation_func(self, dtype):
        if dtype == bool:
            return NotImplemented
        return np.cumsum
