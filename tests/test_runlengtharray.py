from npstructures.runlengtharray import RunLengthArray
from scipy.special import pdtrc
from scipy.stats import poisson
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
import pytest

arrays = [[0, 1, 1, 2, 3, 3, 3],
          [0, 20, 1, 1, 0, 3, 3],
          [2, 2, 2, 1],
          [1],
          [2.3, 19, 32.7, 32.7],
          [2.3, 19, 32.7, 32.7, 92.0, 0.32, 0.84, 0.91],
          [0,1,1,2,2,2,3,3,3,3,2,2,2,1,1,0],
          [True,  True,  True,  True,  True, False, False, False, False,
           False, False,  True,  True,  True,  True,  True,  True,  True,
           True,  True]]

arrays_2 = [[0, 1, 1, 2, 3, 3, 3],
            [10, 1, 1, 2, 3, 2, 2],
            [13, 13, 3, 0],
            [10],
            [32.3, 198, 932.2, 329.7],
            [32.3, 198, 932.2, 329.7, 29.0, 32.0, 34, 19.1]]


@pytest.mark.parametrize("array", arrays)
def test_run_length_array(array):
    array = np.asanyarray(array)
    rlarray = RunLengthArray.from_array(array)
    print(rlarray._values, rlarray._events)
    new_array = np.array(rlarray)
    assert_array_equal(array, new_array)


@pytest.mark.parametrize("array1,array2", tuple(zip(arrays, arrays_2)))
def test_add_run_length_array(array1, array2):
    array1, array2 = (np.asanyarray(a) for a in (array1, array2))
    rl_result = RunLengthArray.from_array(array1)+RunLengthArray.from_array(array2)
    assert_array_equal(np.array(rl_result), array1+array2)

@pytest.mark.parametrize("array1,array2", tuple(zip(arrays, arrays_2)))
def test_maximum_run_length_array(array1, array2):
    array1, array2 = (np.asanyarray(a) for a in (array1, array2))
    rl_result = np.maximum(RunLengthArray.from_array(array1), RunLengthArray.from_array(array2))

    # assert [np.binary_repr(r.view(np.uint64)) for r in rl_result] == [np.binary_repr(r.view(np.uint64)) for r in np.maximum(array1, array2)]
    assert_array_equal(rl_result, np.maximum(array1, array2))


@pytest.mark.parametrize("ufunc", [np.add, np.maximum, pdtrc])
def test_pdtrc_run_length_array(ufunc):
    array1, array2 = (np.asanyarray(a) for a in (arrays[0], arrays_2[0]))
    rl_result = ufunc(RunLengthArray.from_array(array1), RunLengthArray.from_array(array2))

    # assert [np.binary_repr(r.view(np.uint64)) for r in rl_result] == [np.binary_repr(r.view(np.uint64)) for r in np.maximum(array1, array2)]
    assert_array_almost_equal(rl_result, ufunc(array1, array2))



@pytest.mark.parametrize("array", arrays)
def test_add_scalar(array):
    array = np.asanyarray(array)
    res = RunLengthArray.from_array(array) + 10
    assert_array_almost_equal(res, array+10)


@pytest.mark.parametrize("array", arrays)
def test_radd_scalar(array):
    array = np.asanyarray(array)
    res = 10 + RunLengthArray.from_array(array)
    assert_array_almost_equal(res, 10+array)



@pytest.mark.parametrize("array", arrays)
def test_getitem_int(array):
    rlarray = RunLengthArray.from_array(array)
    for i, elem in enumerate(array):
        assert rlarray[i] == elem


@pytest.mark.parametrize("array", arrays)
def test_getitem_slice(array):
    rlarray = RunLengthArray.from_array(array)
    for start, _ in enumerate(array):
        for end in range(start+1, len(array)):
            assert_array_almost_equal(rlarray[start:end], array[start:end])


def _test_ragged_slice():
    vector, starts, ends = (np.asanyarray(e) for e in data)
    starts = np.minimum(starts, ends)
    ends = np.maximum(starts, ends+1)
    subset = vector.view(NPSArray)[starts:ends]
    assert_raggedarray_equal(subset, RaggedArray([vector[start:end] for start, end in zip(starts, ends)]))
