import math

import numpy as np
import pytest
from pyscipopt import Model, cos, exp, log, sin, sqrt
from pyscipopt.scip import Expr, ExprCons, GenExpr, Term


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    return m, x, y, z

CONST = Term()

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
    # sqrt(2) is not a constant expression and
    # we can only power to constant expressions!
    with pytest.raises(NotImplementedError):
        genexpr **= sqrt(2)

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


def test_unary(model):
    m, x, y, z = model

    assert str(abs(x)) == "abs(sum(0.0,prod(1.0,x)))"
    assert str(np.absolute(x)) == "abs(sum(0.0,prod(1.0,x)))"
    assert (
        str(sin([x, y, z]))
        == "[abs(sum(0.0,prod(1.0,x))) abs(sum(0.0,prod(1.0,y))) abs(sum(0.0,prod(1.0,z)))]"
    )
    assert (
        str(np.sin([x, y, z]))
        == "[sin(sum(0.0,prod(1.0,x))) sin(sum(0.0,prod(1.0,y))) sin(sum(0.0,prod(1.0,z)))]"
    )
    assert (
        str(sqrt([x, y, z]))
        == "[sqrt(sum(0.0,prod(1.0,x))) sqrt(sum(0.0,prod(1.0,y))) sqrt(sum(0.0,prod(1.0,z)))]"
    )
    assert (
        str(np.sqrt([x, y, z]))
        == "[sqrt(sum(0.0,prod(1.0,x))) sqrt(sum(0.0,prod(1.0,y))) sqrt(sum(0.0,prod(1.0,z)))]"
    )


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
