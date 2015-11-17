import pytest

from pyscipopt.scip import Model
from pyscipopt.linexpr import LinExpr

m = Model()
x = m.addVar("x")
y = m.addVar("y")
z = m.addVar("z")

def test_operations():
    assert x == x
    assert x != y
    assert x < y or y < x

    expr = x + y
    assert isinstance(expr, LinExpr)
    assert expr[x] == 1.0
    assert expr[y] == 1.0
    assert expr[z] == 0.0

    expr = -x
    assert isinstance(expr, LinExpr)
    assert expr[x] == -1.0
    assert expr[y] ==  0.0

    expr = 4*x
    assert isinstance(expr, LinExpr)
    assert expr[x] == 4.0
    assert expr[y] == 0.0

    expr = x + y + x
    assert isinstance(expr, LinExpr)
    assert expr[x] == 2.0
    assert expr[y] == 1.0

    expr = x + y - x
    assert isinstance(expr, LinExpr)
    assert expr[x] == 0.0
    assert expr[y] == 1.0

    expr = 3*x + 1.0
    assert isinstance(expr, LinExpr)
    assert expr[x] == 3.0
    assert expr[y] == 0.0
    assert expr[()] == 1.0

    expr = 1.0 + 3*x
    assert isinstance(expr, LinExpr)
    assert expr[x] == 3.0
    assert expr[y] == 0.0
    assert expr[()] == 1.0

    with pytest.raises(NotImplementedError):
        expr = x*y

    with pytest.raises(NotImplementedError):
        expr = x*(1 + y)
