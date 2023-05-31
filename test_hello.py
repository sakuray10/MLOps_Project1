from hello import add


def test_add():
    assert 2 == add(1,1)
    assert 5 == add(2,3)