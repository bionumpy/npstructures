from npstructures.raggedshape import RaggedShape, RaggedView
#import npstructures
#import numpy as np
from tests.npbackend import np
import pytest

@pytest.mark.cupy
def test__init__():
    # shape = RaggedShape([0, 3, 5, 7, 11])

    print("name:",np.__name__)
    assert 1==0
    
    shape = RaggedShape([3, 2, 2, 4])
    #if raggedshape.np.__name__ == "cupy":
        #print("CUPY!!!!")
        #np = cp
    assert np.all(shape.starts == np.asanyarray([0, 3, 5, 7]))
    assert np.all(shape.ends == np.asanyarray([3, 5, 7, 11]))
    assert np.all(shape.lengths == np.asanyarray([3, 2, 2, 4]))

@pytest.mark.cupy
def test_empty_view():
    view = RaggedView(np.array([], dtype=np.int32))
    shape = view.get_shape()
    assert shape == RaggedShape(np.array([], dtype=np.int32), is_coded=True)

