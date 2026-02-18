import math

import numpy as np
import pytest

from pyscipopt import Model, cos, exp, log, sin, sqrt
from pyscipopt.scip import CONST, Expr, ExprCons, GenExpr


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    return m, x, y, z


def test_upgrade(model):
    m, x, y, z = model
    expr = x + y
    assert isinstance(expr, Expr)
    expr += exp(z)
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    expr -= exp(z)
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    expr /= x
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    expr *= sqrt(x)
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    expr **= 1.5
    assert isinstance(expr, GenExpr)

    expr = x + y
    assert isinstance(expr, Expr)
    assert isinstance(expr + exp(x), GenExpr)
    assert isinstance(expr - exp(x), GenExpr)
    assert isinstance(expr/x, GenExpr)
    assert isinstance(expr * x**1.2, GenExpr)
    assert isinstance(sqrt(expr), GenExpr)
    assert isinstance(abs(expr), GenExpr)
    assert isinstance(log(expr), GenExpr)
    assert isinstance(exp(expr), GenExpr)
    assert isinstance(sin(expr), GenExpr)
    assert isinstance(cos(expr), GenExpr)

    with pytest.raises(ZeroDivisionError):
        expr /= 0.0

def test_genexpr_op_expr(model):
    m, x, y, z = model
    genexpr = x**1.5 + y
    assert isinstance(genexpr, GenExpr)
    genexpr += x**2
    assert isinstance(genexpr, GenExpr)
    genexpr += 1
    assert isinstance(genexpr, GenExpr)
    genexpr += x
    assert isinstance(genexpr, GenExpr)
    genexpr += 2 * y
    assert isinstance(genexpr, GenExpr)
    genexpr -= x**2
    assert isinstance(genexpr, GenExpr)
    genexpr -= 1
    assert isinstance(genexpr, GenExpr)
    genexpr -= x
    assert isinstance(genexpr, GenExpr)
    genexpr -= 2 * y
    assert isinstance(genexpr, GenExpr)
    genexpr *= x + y
    assert isinstance(genexpr, GenExpr)
    genexpr *= 2
    assert isinstance(genexpr, GenExpr)
    genexpr /= 2
    assert isinstance(genexpr, GenExpr)
    genexpr /= x + y
    assert isinstance(genexpr, GenExpr)
    assert isinstance(x**1.2 + x + y, GenExpr)
    assert isinstance(x**1.2 - x, GenExpr)
    assert isinstance(x**1.2 *(x+y), GenExpr)

def test_genexpr_op_genexpr(model):
    m, x, y, z = model
    genexpr = x**1.5 + y
    assert isinstance(genexpr, GenExpr)
    genexpr **= 2.2
    assert isinstance(genexpr, GenExpr)
    genexpr += exp(x)
    assert isinstance(genexpr, GenExpr)
    genexpr -= exp(x)
    assert isinstance(genexpr, GenExpr)
    genexpr /= log(x + 1)
    assert isinstance(genexpr, GenExpr)
    genexpr *= (x + y)**1.2
    assert isinstance(genexpr, GenExpr)
    genexpr /= exp(2)
    assert isinstance(genexpr, GenExpr)
    genexpr /= x + y
    assert isinstance(genexpr, GenExpr)
    genexpr = x**1.5 + y
    assert isinstance(genexpr, GenExpr)
    assert isinstance(sqrt(x) + genexpr, GenExpr)
    assert isinstance(exp(x) + genexpr, GenExpr)
    assert isinstance(sin(x) + genexpr, GenExpr)
    assert isinstance(cos(x) + genexpr, GenExpr)
    assert isinstance(1/x + genexpr, GenExpr)
    assert isinstance(1/x**1.5 - genexpr, GenExpr)
    assert isinstance(y/x - exp(genexpr), GenExpr)

    genexpr **= sqrt(2)
    assert isinstance(genexpr, GenExpr)

    with pytest.raises(TypeError):
        genexpr **= sqrt("2")

def test_degree(model):
    m, x, y, z = model
    expr = GenExpr()
    assert expr.degree() == float('inf')

# In contrast to Expr inequalities, we can't expect much of the sides
def test_inequality(model):
    m, x, y, z = model

    expr = x + 2*y
    assert isinstance(expr, Expr)
    cons = expr <= x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, GenExpr)
    assert cons._lhs is None
    assert cons._rhs == 0.0

    assert isinstance(expr, Expr)
    cons = expr >= x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, GenExpr)
    assert cons._lhs == 0.0
    assert cons._rhs is None

    assert isinstance(expr, Expr)
    cons = expr >= 1 + x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, GenExpr)
    assert cons._lhs == 0.0 # NOTE: the 1 is passed to the other side because of the way GenExprs work
    assert cons._rhs is None

    assert isinstance(expr, Expr)
    cons = exp(expr) <= 1 + x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, GenExpr)
    assert cons._rhs == 0.0
    assert cons._lhs is None


def test_equation(model):
    m, x, y, z = model
    equat = 2*x**1.2 - 3*sqrt(y) == 1
    assert isinstance(equat, ExprCons)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 1.0

    equat = exp(x+2*y) == 1 + x**1.2
    assert isinstance(equat, ExprCons)
    assert isinstance(equat.expr, GenExpr)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 0.0

    equat = x == 1 + x**1.2
    assert isinstance(equat, ExprCons)
    assert isinstance(equat.expr, GenExpr)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 0.0

def test_rpow_constant_base(model):
    m, x, y, z = model
    a = 2**x
    b = exp(x * log(2.0))
    assert isinstance(a, GenExpr)
    assert repr(a) == repr(b) # Structural equality is not implemented; compare strings
    m.addCons(2**x <= 1)

    with pytest.raises(ValueError):
        c = (-2)**x


def test_getVal_with_GenExpr():
    m = Model()
    x = m.addVar(lb=1, ub=1, name="x")
    y = m.addVar(lb=2, ub=2, name="y")
    z = m.addVar(lb=0, ub=0, name="z")
    m.optimize()

    # test "Expr({Term(x, y, z): 1.0})"
    assert m.getVal(z * x * y) == 0
    # test "Expr({Term(x): 1.0, Term(y): 1.0, Term(): 1.0})"
    assert m.getVal(x + y + 1) == 4
    # test "prod(1.0,sum(0.0,prod(1.0,x)),**(sum(0.0,prod(1.0,x)),-1))"
    assert m.getVal(x / x) == 1
    # test "prod(1.0,sum(0.0,prod(1.0,y)),**(sum(0.0,prod(1.0,x)),-1))"
    assert m.getVal(y / x) == 2
    # test "**(prod(1.0,**(sum(0.0,prod(1.0,x)),-1)),2)"
    assert m.getVal((1 / x) ** 2) == 1
    # test "sin(sum(0.0,prod(1.0,x)))"
    assert round(m.getVal(sin(x)), 6) == round(math.sin(1), 6)

    with pytest.raises(TypeError):
        m.getVal(1)

    with pytest.raises(ZeroDivisionError):
        m.getVal(1 / z)


def test_unary(model):
    m, x, y, z = model

    res = "abs(sum(0.0,prod(1.0,x)))"
    assert str(abs(x)) == res
    assert str(np.absolute(x)) == res

    res = "[sin(sum(0.0,prod(1.0,x))) sin(sum(0.0,prod(1.0,y)))]"
    assert str(sin([x, y])) == res
    assert str(np.sin([x, y])) == res

    res = "[cos(sum(0.0,prod(1.0,x))) cos(sum(0.0,prod(1.0,y)))]"
    assert str(cos([x, y])) == res
    assert str(np.cos([x, y])) == res

    res = "[sqrt(sum(0.0,prod(1.0,x))) sqrt(sum(0.0,prod(1.0,y)))]"
    assert str(sqrt([x, y])) == res
    assert str(np.sqrt([x, y])) == res

    res = "[exp(sum(0.0,prod(1.0,x))) exp(sum(0.0,prod(1.0,y)))]"
    assert str(exp([x, y])) == res
    assert str(np.exp([x, y])) == res

    res = "[log(sum(0.0,prod(1.0,x))) log(sum(0.0,prod(1.0,y)))]"
    assert str(log([x, y])) == res
    assert str(np.log([x, y])) == res

    assert sqrt(4) == np.sqrt(4)
    assert all(sqrt([4, 4]) == np.sqrt([4, 4]))
    assert exp(3) == np.exp(3)
    assert all(exp([3, 3]) == np.exp([3, 3]))
    assert log(5) == np.log(5)
    assert all(log([5, 5]) == np.log([5, 5]))
    assert sin(1) == np.sin(1)
    assert all(sin([1, 1]) == np.sin([1, 1]))
    assert cos(1) == np.cos(1)
    assert all(cos([1, 1]) == np.cos([1, 1]))

    # test invalid unary operations
    with pytest.raises(TypeError):
        np.arcsin(x)


def test_mul():
    m = Model()
    x = m.addVar(name="x")
    y = m.addVar(name="y")

    assert str(Expr({CONST: 1.0}) * x) == "Expr({Term(x): 1.0})"
    assert str(y * Expr({CONST: -1.0})) == "Expr({Term(y): -1.0})"
    assert str((x - x) * y) == "Expr({Term(x, y): 0.0})"
    assert str(y * (x - x)) == "Expr({Term(x, y): 0.0})"
    assert (
        str((x + 1) * (y - 1))
        == "Expr({Term(x, y): 1.0, Term(x): -1.0, Term(y): 1.0, Term(): -1.0})"
    )
    assert (
        str((x + 1) * (x + 1) * y)
        == "Expr({Term(x, x, y): 1.0, Term(x, y): 2.0, Term(y): 1.0})"
    )


def test_abs_abs_expr():
    m = Model()
    x = m.addVar(name="x")

    # should print abs(x) not abs(abs(x))
    assert str(abs(abs(x))) == str(abs(x))
