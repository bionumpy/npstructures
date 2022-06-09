import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as stnp
from hypothesis.strategies import composite


def array_is_valid(a):
    # should return false for arrays with too large elements
    if len(a.ravel()) == 0:
        return True

    return (
        a.dtype in [bool] or
        np.max(a) < np.iinfo(a.dtype).max // len(a)
    )


def array_shape_is_valid(shape):
    if len(shape) == 0 or len(shape) == 1:
        return True

    if sum(shape) == 0:
        return True

    return shape[0] != 0  # first dimension cannot be 0 if not all are


@composite
def integers(draw):
    # wrapper function to remove integers larger than what numpy can handle
    return draw(integers().filter(lambda i: i < np.iinfo(np.int64).max))


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
def matrices(draw, arrays=arrays()):
    return draw(arrays)


@composite
def matrix_and_indexes(draw, matrices=matrices()):
    m = draw(matrices)
    indexes = draw(stnp.basic_indices(m.shape))
    return m, indexes


@composite
def single_lists(draw, elements=integers(), min_size=0):
    return draw(st.lists(elements, min_size=min_size))


@composite
def nested_lists(draw, elements=single_lists(), min_size=0):
    return draw(st.lists(elements, min_size=min_size))


@composite
def list_of_arrays(draw):
    return draw(nested_lists(arrays(stnp.integer_dtypes(), array_shapes(0, 1, 1))))


@composite
def nonempty_list_of_arrays(draw):
    return draw(nested_lists(arrays(stnp.integer_dtypes(), array_shapes(0, 1, 1)), min_size=1))


@composite
def two_nested_lists(draw):
    return [draw(nested_lists(min_size=1)) for _ in range(2)]


if __name__ == "__main__":
    print(two_nested_lists().example())
