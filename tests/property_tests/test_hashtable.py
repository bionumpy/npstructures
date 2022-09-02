import pytest
from numpy import array, int8, int16, int32, int64
from numpy.testing import assert_equal
from npstructures import HashTable
import hypothesis.extra.numpy as stnp
from hypothesis import given, example
from .strategies import matrix_and_indexes, matrices, nested_lists, arrays, array_shapes, two_nested_lists, integers
from hypothesis.strategies import composite
import hypothesis.strategies as st


modulos = [None, 21, 20033, 1000]

@composite
def hashtable_keys(draw):
    shape = draw(stnp.array_shapes(min_dims=1, max_dims=1, min_side=1))
    keys = draw(stnp.arrays(shape=shape,
                            dtype=stnp.integer_dtypes(),
                            unique=True)
                )
    return keys


@composite
def hashtable_keys_and_values(draw):
    keys = draw(hashtable_keys())
    values = draw(stnp.arrays(shape=keys.shape, dtype=stnp.integer_dtypes()))
    return keys, values


@composite
def hashtable_keys_and_two_values(draw):
    keys, values = draw(hashtable_keys_and_values())
    values2 = draw(stnp.arrays(shape=values.shape,
                               dtype=values.dtype))
    return keys, values, values2


@given(data=hashtable_keys_and_values(), single_value=integers(), mod=st.sampled_from(modulos))
def test_lookup_int(data, single_value, mod):
    keys, values = data
    h = HashTable(keys, single_value, mod)
    assert_equal(h[keys], single_value)


@given(data=hashtable_keys_and_values(), mod=st.sampled_from(modulos))
def test_lookup(data, mod):
    keys, values = data
    h = HashTable(keys, values, mod=mod)
    assert_equal(h[keys], values)


@given(data=hashtable_keys_and_values(), single_value=integers(), mod=st.sampled_from(modulos))
@example(data=(array([0], dtype=int8), array([0], dtype=int8)),
         single_value=128,
         mod=None)
def test_set_all_to_single_value(data, single_value, mod):
    keys, values = data
    h = HashTable(keys, values, mod=mod)
    h[keys] = single_value
    assert_equal(h[keys], values.dtype.type(single_value))


@given(data=hashtable_keys_and_values(), single_value=integers(), mod=st.sampled_from(modulos))
def test_set_some_to_single_value(data, single_value, mod):
    keys, values = data
    h = HashTable(keys, values, mod=mod)
    # change half of the values
    change_keys = keys[::2]
    h[change_keys] = single_value
    assert_equal(h[change_keys], values.dtype.type(single_value))


@given(data=hashtable_keys_and_two_values(), mod=st.sampled_from(modulos))
def test_set_to_multiple_values(data, mod):
    keys, values, values2 = data
    h = HashTable(keys, values, mod=mod)
    h[keys] = values2
    assert_equal(h[keys], values2)
    h[keys] = values
    assert_equal(h[keys], values)
    # change half keys
    h[keys[::2]] = values2[::2]
    assert_equal(h[keys[::2]], values2[::2])


if __name__ == "__main__":
    print(hashtable_keys_and_values().example())
