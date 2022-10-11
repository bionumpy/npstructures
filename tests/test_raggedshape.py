from npstructures.raggedshape import RaggedShape, RaggedView
from tests.npbackend import np
import pytest

@pytest.mark.cupy
def test__init__():
    # shape = RaggedShape([0, 3, 5, 7, 11])
    
    shape = RaggedShape([3, 2, 2, 4])
    np.testing.assert_equal(shape.starts, [0, 3, 5, 7])
    np.testing.assert_equal(shape.ends, [3, 5, 7, 11])
    np.testing.assert_equal(shape.lengths, [3, 2, 2, 4])

@pytest.mark.cupy
def test_empty_view():
    view = RaggedView(np.array([], dtype=np.int32))
    shape = view.get_shape()
    assert shape == RaggedShape(np.array([], dtype=np.int32), is_coded=True)

