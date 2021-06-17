"""Unit testing for module."""
import pytest
from spatialHeterogeneity.module import Foo


@pytest.mark.parametrize(
    "the_argument",
    [(0), (10 ** 6), (42)],
)
def test_foo(the_argument):
    """Pytest example"""
    assert (
        "works"
        in Foo(a=the_argument).method_that_would_really_waste_your_time_if_it_fails()
    )


def test_make_foo_fail():
    """Never trust a test that doesn't fail"""
    foo = Foo(1.0)  # type: ignore
    with pytest.raises(TypeError):
        foo.method_that_would_really_waste_your_time_if_it_fails()


# fixtures are another great feature of pytest, that allow arrangement of the test and cleanup after
