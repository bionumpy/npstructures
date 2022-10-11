from tests.npbackend import np
import pytest
from npstructures import RaggedArray, Counter, RaggedShape, HashTable
import collections

do_run = False
rng = np.random.default_rng(129839)


@pytest.fixture
def random_array():
    shape = tuple(np.random.randint(1, 100, 2))
    return np.random.rand(shape[0] * shape[1]).reshape(shape)


@pytest.mark.skipif(not do_run, reason="random")
def test_cumsum(random_array):
    ra = RaggedArray.from_numpy_array(random_array)
    cm = np.cumsum(ra, axis=-1)
    assert np.allclose(cm.to_numpy_array(), np.cumsum(random_array, axis=-1))


@pytest.mark.parametrize("reduction", ["sum", "mean", "std", "max", "min"])
@pytest.mark.skipif(not do_run, reason="random")
def test_reduction(random_array, reduction):
    ra = RaggedArray.from_numpy_array(random_array)
    res = getattr(ra, reduction)(axis=-1)
    true = getattr(random_array, reduction)(axis=-1)
    assert np.allclose(res, true)


def get_key_sample_pairs():
    p = 6
    for n_keys in (10 ** i for i in range(1, p)):
        for n_samples in (10 ** i for i in range(1, p)):
            keys = np.cumsum(rng.integers(1, 100, size=n_keys))
            samples = rng.choice(keys, size=n_samples)
            yield keys, samples


# @pytest.mark.parametrize("keys,samples", get_key_sample_pairs())
@pytest.mark.skipif(not do_run, reason="random")
def test_counter():
    for keys, samples in get_key_sample_pairs():
        c = Counter(keys, mod=100)
        c.count(samples)
        true = collections.Counter(samples)
        assert all(true[key] == val for key, val in c.items())


@pytest.mark.skipif(not do_run, reason="random")
def test_counter2():
    n_keys = 100000
    keys = np.cumsum(rng.integers(1, 100, size=n_keys))
    mod = 997
    for i in range(7):
        counts = rng.integers(0, 100, size=n_keys)
        shape = RaggedShape(counts)
        samples = shape.broadcast_values(keys[:, None])
        rng.permuted(samples, out=samples)
        c = Counter(keys, mod=mod)
        c.count(samples)
        assert c == HashTable(keys, counts, mod=mod)
