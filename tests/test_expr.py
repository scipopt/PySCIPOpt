import pytest

from pyscipopt import Model, sqrt, log, exp, sin, cos
from pyscipopt.scip import Expr, GenExpr, ExprCons, Term, quicksum

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
