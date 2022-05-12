from npstructures.raggedshape import RaggedShape, RaggedView
import numpy as np


def test__init__():
    # shape = RaggedShape([0, 3, 5, 7, 11])
    shape = RaggedShape([3, 2, 2, 4])
    assert np.all(shape.starts == [0, 3, 5, 7])
    assert np.all(shape.ends == [3, 5, 7, 11])
    assert np.all(shape.lengths == [3, 2, 2, 4])


def test_empty_view():
    view = RaggedView(np.array([], dtype=np.int32))
    shape = view.get_shape()
    assert shape == RaggedShape(np.array([], dtype=np.int32), is_coded=True)
