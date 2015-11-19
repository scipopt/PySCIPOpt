import pytest

from pyscipopt.scip import Model
from pyscipopt.linexpr import LinExpr, LinCons

m = Model()
x = m.addVar("x")
y = m.addVar("y")
z = m.addVar("z")

def test_variable():
    assert x < y or y < x

def test_operations():
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

def test_inequality():
    expr = x + 2*y
    cons = expr <= 0
    assert isinstance(cons, LinCons)
    assert cons.lb is None
    assert cons.ub == 0.0
    assert cons.expr[x] == 1.0
    assert cons.expr[y] == 2.0
    assert cons.expr[z] == 0.0
    assert cons.expr[()] == 0.0

    cons = expr >= 5
    assert isinstance(cons, LinCons)
    assert cons.lb == 5.0
    assert cons.ub is None
    assert cons.expr[x] == 1.0
    assert cons.expr[y] == 2.0
    assert cons.expr[z] == 0.0
    assert cons.expr[()] == 0.0

    cons = 5 <= x + 2*y - 3
    assert isinstance(cons, LinCons)
    assert cons.lb == 8.0
    assert cons.ub is None
    assert cons.expr[x] == 1.0
    assert cons.expr[y] == 2.0
    assert cons.expr[z] == 0.0
    assert cons.expr[()] == 0.0

def test_ranged():
    expr = x + 2*y
    cons = expr >= 3
    ranged = cons <= 5
    assert isinstance(ranged, LinCons)
    assert ranged.lb == 3.0
    assert ranged.ub == 5.0
    assert ranged.expr[y] == 2.0
    assert ranged.expr[()] == 0.0

    # again, more or less directly:
    ranged = 3 <= (x + 2*y <= 5)
    assert isinstance(ranged, LinCons)
    assert ranged.lb == 3.0
    assert ranged.ub == 5.0
    assert ranged.expr[y] == 2.0
    assert ranged.expr[()] == 0.0
    # we must use the paranthesis, because
    #     x <= y <= z
    # is a "chained comparison", which will be interpreted by Python
    # to be equivalent to
    #     (x <= y) and (y <= z)
    # where "and" can not be overloaded and the expressions in
    # paranthesis are coerced to booleans.

    with pytest.raises(TypeError):
        ranged = (x + 2*y <= 5) <= 3

    with pytest.raises(TypeError):
        ranged = 3 >= (x + 2*y <= 5)

def test_equation():
    equat = 2*x - 3*y == 1
    assert isinstance(equat, LinCons)
    assert equat.lb == equat.ub
    assert equat.lb == 1.0
    assert equat.expr[x] == 2.0
    assert equat.expr[y] == -3.0
    assert equat.expr[()] == 0.0
