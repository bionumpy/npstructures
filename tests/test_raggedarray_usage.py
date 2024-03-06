import pytest

from npstructures import RaggedArray

# this works
@pytest.mark.xfail
def test_assert_raises_value_error():
    with pytest.raises(ValueError):
        x = RaggedArray([0, 1, 2, 3], [4, 5, 6])


