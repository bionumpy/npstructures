import numpy as np
from .raggedarray import RaggedArray
import dataclasses


def shallow_tuple(self):
    return tuple(
        getattr(self, field.name) for field in dataclasses.fields(self)
    )


def assert_npdataclass_equal(a, b):
    assert dataclasses.fields(a) == dataclasses.fields(b)
    
    for s, o, field in zip(shallow_tuple(a), shallow_tuple(b), dataclasses.fields(a)):
        assert hasattr(s, "shape") and hasattr(o, "shape"), (s, o)
        if not s.shape == o.shape:
            assert False, (s.shape, o.shape, str(a), str(b))
        if not np.all(np.equal(s, o)):
            assert False, (a, b, field.name)


def assert_raggedarray_equal(a, b):
    if isinstance(a, list):
        a = RaggedArray(a)

    if isinstance(b, list):
        b = RaggedArray(b)

    assert len(a.shape) == len(b.shape), (str(a), str(b))
    assert np.all(a.shape[-1] == b.shape[-1]), (a.shape, b.shape)
    assert a.shape[:-1] == b.shape[:-1]

    # assert a._shape == b._shape, (str(a), str(b))
    assert np.all(a.ravel() == b.ravel()), (str(a), str(b))
