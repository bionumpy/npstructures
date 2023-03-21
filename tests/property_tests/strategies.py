from tests.npbackend import np
import hypothesis.strategies as st
import hypothesis.extra.numpy as stnp
from hypothesis.strategies import composite


def array_is_valid(a):
    # should return false for arrays with too large elements
    if len(a.ravel()) == 0:
        return True
    t = not (np.issubdtype(a.dtype, np.integer) and (np.max(a) >= (np.iinfo(a.dtype).max-1)))
    t &= not (np.issubdtype(a.dtype, np.integer) and (np.min(a) <= (np.iinfo(a.dtype).min+1)))
    t &= not (np.issubdtype(a.dtype, np.floating) and (np.any(np.isinf(a) | np.isnan(a) | (np.abs(a) > 10**37))))
    return t




def array_shape_is_valid(shape):
    if len(shape) == 0 or len(shape) == 1:
        return True

    if sum(shape) == 0:
        return True

    return shape[0] != 0  # first dimension cannot be 0 if not all are



@composite
def integers(draw):
    # wrapper function to remove integers larger than what numpy can handle
    max_value = np.iinfo(np.int64).max
    min_value = np.iinfo(np.int64).min
    return draw(st.integers(min_value=min_value, max_value=max_value))


@composite
def array_shapes(draw, min_side=0, min_dims=2, max_dims=2):
    return draw(stnp.array_shapes(min_side=min_side,
                                  min_dims=min_dims,
                                  max_dims=max_dims).filter(array_shape_is_valid))


@composite
def arrays(draw, dtype=stnp.integer_dtypes(), array_shape=array_shapes(0, 2, 2)):
    return draw(stnp.arrays(
        dtype,
        array_shape
    ).filter(array_is_valid))


@composite
def two_arrays(draw, dtype=stnp.integer_dtypes() | stnp.floating_dtypes(), array_shape=array_shapes(0, 2, 2)):
    shape = draw(array_shape)
    return draw(arrays(dtype, shape)), draw(arrays(dtype, shape))


@composite
def array_and_column(draw, dtype=stnp.integer_dtypes() | stnp.floating_dtypes(), array_shape=array_shapes(0, 2, 2)):
    shape = draw(array_shape)
    return draw(arrays(dtype, shape)), draw(arrays(dtype, (shape[0], 1)))


@composite
def matrices(draw, arrays=arrays()):
    return draw(arrays)


@composite
def vector_and_indexes(draw):
    shape = draw(array_shapes(1, 1, 1))
    m = draw(arrays(array_shape=shape))
    indexes = draw(stnp.basic_indices(m.shape) | raw_boolean_indices((m.shape[0],)))
    return m, indexes


@composite
def vector_and_startends(draw):
    shape = draw(array_shapes(1, 1, 1))
    m = draw(arrays(array_shape=shape))
    starts = draw(st.lists(st.integers(min_value=0, max_value=shape[0]-1), min_size=1))
    ends = draw(st.lists(st.integers(min_value=0, max_value=shape[0]-1), min_size=len(starts), max_size=len(starts)))
    return m, starts, ends

@composite
def matrix_and_indexes(draw, matrices=matrices()):
    m = draw(matrices)
    indexes = draw(stnp.basic_indices(m.shape) | raw_boolean_indices((m.shape[0],)))
    return m, indexes


def matrix_and_boolean(draw, matrices=matrices()):
    m = draw(matrices)
    boolean = draw(stnp.arrays(shape=m.shape, dtype=bool))
    return m, boolean


@composite
def matrix_and_row_indexes(draw, matrices=matrices()):
    m = draw(matrices)
    indexes = draw(stnp.basic_indices((m.shape[0],), allow_ellipsis=False) | raw_boolean_indices((m.shape[0],)))
    return m, indexes


@composite
def matrix_and_indexes_and_values(draw, matrices=matrices()):
    m = draw(matrices)
    indexes = draw((stnp.basic_indices(m.shape) | raw_boolean_indices((m.shape[0],))).filter(lambda i: m[i].size>0))
    values = draw(stnp.arrays(dtype=m.dtype, shape=m[indexes].shape))
    return m, indexes, values


@composite
def boolean_indices(draw, shape):
    row_indices = draw(stnp.arrays(shape=(shape[0], ), dtype=bool))
    col_indices = draw((stnp.integer_array_indices((shape[1],)) | stnp.basic_indices((shape[1],))).filter(lambda x: not isinstance(x, tuple)))
    return (row_indices, col_indices)


def raw_boolean_indices(shape):
    return stnp.arrays(shape=(shape[0], ), dtype=bool)


@composite
def matrix_and_integer_array_indexes(draw, matrices=arrays(array_shape=array_shapes(1, 2, 2))):
    m = draw(matrices)
    indexes = draw(stnp.integer_array_indices(m.shape) | boolean_indices(m.shape))
    return m, indexes


@composite
def single_lists(draw, elements=integers(), min_size=0):
    return draw(st.lists(elements, min_size=min_size))


@composite
def nested_lists(draw, elements=single_lists(), min_size=0):
    return draw(st.lists(elements, min_size=min_size))


@composite
def list_of_arrays(draw, min_size=0, min_length=0, dtypes=stnp.integer_dtypes() | stnp.floating_dtypes().filter(lambda x: x != np.float16)):
    dtype = draw(dtypes)
    return draw(nested_lists(arrays(dtype, array_shapes(min_length, 1, 1)), min_size=min_size))


@composite
def nonempty_list_of_arrays(draw):
    return draw(nested_lists(arrays(stnp.integer_dtypes(), array_shapes(0, 1, 1)), min_size=1))


@composite
def two_nested_lists(draw):
    return [draw(nested_lists(min_size=1)) for _ in range(2)]


@composite
def ragged_indices(draw, lens):
    return [draw(st.integers(0, l)) for l in lens]


@composite
def nested_list_and_indices(draw):
    nl = draw(nested_lists(elements=single_lists(min_size=1), min_size=1))
    row_indices = draw(stnp.integer_array_indices((len(nl),), result_shape=stnp.array_shapes(min_dims=1, max_dims=1)))
    col_indices = st.builds(lambda *args: list(args), *(st.integers(0, len(nl[i])-1) for i in row_indices[0]))
    return nl, row_indices[0], draw(col_indices)


@composite
def nested_list_and_slices(draw):
    nl = draw(nested_lists(elements=single_lists(min_size=1), min_size=1))
    row_indices = draw(stnp.basic_indices(shape=(len(nl),), allow_ellipsis=False).filter(lambda x: not isinstance(x, tuple)))
    max_len = max(len(l) for l in nl)
    col_indices = draw(stnp.basic_indices((max_len,), allow_ellipsis=False).filter(lambda x: isinstance(x, slice)))
    assert isinstance(col_indices, slice)
    return nl, row_indices, col_indices


if __name__ == "__main__":
    print(two_nested_lists().example())
