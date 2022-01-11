from npstructures.numpylist import NumpyList
import numpy as np

def test():
    list = NumpyList()
    list.append(5.0)
    list.append(10.0)

    assert np.all(list.get_nparray() == [5.0, 10.0])

    list2 = NumpyList(dtype=np.uint32)
    for i in range(10000):
        list2.append(i)

    array = list2.get_nparray()
    assert array.dtype == np.uint32
    assert len(array) == 10000

    print(array)


test()