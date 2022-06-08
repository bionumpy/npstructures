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


@composite
def arrays(draw, dtype=stnp.integer_dtypes(), min_size=0, min_dims=1, max_dims=1):
    return draw(stnp.arrays(
        dtype,
        stnp.array_shapes(min_side=min_size, min_dims=min_dims, max_dims=max_dims),
    ).filter(array_is_valid))


@composite
def matrices(draw, dtype=stnp.scalar_dtypes()):
    return draw(arrays(dtype, min_dims=2, max_dims=2))


@composite
def matrix_and_indexes(draw, matrices=matrices()):
    m = draw(matrices)
    indexes = draw(stnp.basic_indices(m.shape))
    return m, indexes


@composite
def single_lists(draw, elements=st.integers(), min_size=0):
    return draw(st.lists(elements, min_size=min_size))


@composite
def nested_lists(draw, elements=single_lists(), min_size=0):
    return draw(st.lists(elements, min_size=min_size))


if __name__ == "__main__":
    print(arrays(min_dims=2, max_dims=2).example())

