import numpy as np


class NpWrapper:
    def __init__(self):
        self._backend = np

    def __getattr__(self, item):
        return getattr(self._backend, item)

    def set_backend(self, lib):
        print("Setting backend to %s" % lib)
        self._backend = lib

    def __dir__(self):
        return self._backend.__dir__()

    @property
    def __dict__(self):
        return self._backend.__dict__


np = NpWrapper()


def as_strided(array, shape=None, strides=None, **kwargs):
    if strides is None:
        assert len(array.shape) == 1
        if len(shape) == 2:
            strides = (shape[-1]*array.strides[-1], array.strides[-1])
        elif len(shape) == 1:
            strides = (array.strides[-1],)
        else:
            assert False, (array, shape, len(shape))

    if hasattr(array, "as_strided"):
        return array.as_strided(shape, strides, **kwargs)
    assert not np.issubdtype(array.dtype, np.object_)
    return np.lib.stride_tricks.as_strided(array, shape, strides, **kwargs)


def unsafe_extend_right(array, n=1):
    assert len(array.shape) == 1, array.shape
    return np.append(array, np.zeros_like(array, shape=(n, )))
    # return as_strided(array, shape=(array.size+n, ), writeable=False, subok=True)


def unsafe_extend_right_2d(array, n=1):
    assert len(array.shape) == 2, array.shape
    return np.concatenate((array, np.zeros_like(array, shape=(array.shape[0], n))), axis=-1)
    # return as_strided(array, shape=(array.shape[0], array.shape[1]+n), writeable=False, subok=True)


def unsafe_extend_left(array, n=1):
    assert len(array.shape) == 1, array.shape
    return np.insert(array, 0, np.zeros_like(array, shape=(n,)))
    # return unsafe_extend_right(array[::-1], n)[::-1]


def unsafe_extend_left_2d(array, n=1):
    assert len(array.shape) == 2
    return np.concatenate((np.zeros_like(array, shape=(array.shape[0], n)), array), axis=-1)
    # return unsafe_extend_right_2d(array[:, ::-1], n)[:, ::-1]


def bincount2d(array):
    max_val = np.max(array)+1
    n_rows, n_cols = array.shape
    counts = np.zeros((n_rows, max_val), dtype=int)
    np.add.at(counts, (np.broadcast_to(np.arange(n_rows)[:, np.newaxis], array.shape), array), 1)
    return counts
