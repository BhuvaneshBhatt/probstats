import sympy as sp

from probstats import RandomVariable, depends_on, random_variables


def test_random_variable_is_native_symbolic_atom():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    a = sp.Symbol("a")

    expression = 2 * x + a * y**2 - 3

    assert isinstance(x, sp.Symbol)
    assert sp.expand(expression - (2 * x + a * y**2 - 3)) == 0
    assert random_variables(expression) == frozenset({x, y})
    assert expression.free_symbols == {x, y, a}


def test_random_variable_survives_symbolic_transformations():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    f = sp.Function("f")

    expression = sp.exp(x) + f(x * y)

    assert random_variables(expression) == frozenset({x, y})
    assert depends_on(expression, x)
    assert depends_on(expression, (y,))


def test_deterministic_symbols_are_not_random_variables():
    x = RandomVariable("X")
    a = sp.Symbol("a")

    assert random_variables(a * x) == frozenset({x})
    assert random_variables(a + 1) == frozenset()
    assert not depends_on(a + 1, x)


def test_random_variable_validation_and_symbol_construction():
    source = sp.Symbol("X", real=True)
    x = RandomVariable(source)

    assert x.name == "X"
    assert x.is_real is True

    for bad in ("", 3, None):
        try:
            RandomVariable(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"accepted invalid random-variable name {bad!r}")
