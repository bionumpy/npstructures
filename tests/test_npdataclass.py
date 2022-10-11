import pytest
from tests.npbackend import np
from npstructures.npdataclasses import VarLenArray, npdataclass


@npdataclass
class DummyClass:
    data1: np.ndarray
    data2: np.ndarray


@npdataclass
class DummyClass2:
    data2: np.ndarray


@pytest.fixture
def objects():
    return [
        DummyClass(np.arange(4), 2 * np.arange(4)),
        DummyClass(np.arange(3) + 2, np.arange(3)),
    ]


@pytest.fixture
def objects2():
    return [DummyClass2(2 * np.arange(4)), DummyClass2(np.arange(3))]


def test_getitem(objects):
    assert objects[0][2] == DummyClass.single_entry(2, 4)
    assert objects[0][:2] == DummyClass([0, 1], [0, 2])


def test_concat(objects):
    assert np.concatenate(objects) == DummyClass(
        np.concatenate((np.arange(4), np.arange(3) + 2)),
        np.concatenate((2 * np.arange(4), np.arange(3))),
    )


def test_astype(objects, objects2):
    for o, o2 in zip(objects, objects2):
        assert o.astype(DummyClass2) == o2


def test_varlenarray():
    a1 = VarLenArray(np.arange(6).reshape(3, 2))
    a2 = VarLenArray(np.arange(3).reshape(3, 1))
    v = np.concatenate((a1, a2))
    assert np.all(v.array == np.concatenate((a1.array, [[0, 0], [0, 1], [0, 2]])))
