import numpy as np
from .raggedarray import RaggedArray
import dataclasses


def shallow_tuple(self):
    return tuple(
        getattr(self, field.name) for field in dataclasses.fields(self)
    )


def assert_npdataclass_equal(a, b):
    print(a)
    print(b)
    for s, o in zip(shallow_tuple(a), shallow_tuple(b)):
        assert hasattr(s, "shape") and hasattr(o, "shape"), (s, o)
        if not s.shape == o.shape:
            assert False, (s.shape, o.shape, str(a), str(b))
        if not np.all(np.equal(s, o)):
            assert False, (a, b)


def assert_raggedarray_equal(a, b):
    if isinstance(a, list):
        a = RaggedArray(a)

    if isinstance(b, list):
        b = RaggedArray(b)

    assert a.shape == b.shape, (str(a), str(b))
    assert np.all(a.ravel() == b.ravel()), (str(a), str(b))
