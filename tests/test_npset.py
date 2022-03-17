from npstructures.npset import NpSet


def simple_test():
    s = NpSet([1, 5, 3, 100, 200])

    assert 1 in s
    assert 2 not in s
    assert 200 in s
    assert 100 in s


simple_test()