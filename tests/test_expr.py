import pytest

from pyscipopt import Model, cos, exp, log, sin, sqrt
from pyscipopt.scip import AbsExpr, Expr, ExprCons, Term

CONST = Term()


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    return m, x, y, z


def test_Expr_init_error():
    with pytest.raises(TypeError):
        Expr({42: 1})

    with pytest.raises(TypeError):
        Expr({"42": 0})

    x = Model().addVar("x")
    with pytest.raises(TypeError):
        Expr({x: 42})


def test_Expr_slots():
    x = Model().addVar("x")
    t = Term(x)
    e = Expr({t: 1.0})

    # Verify we can access defined slots/attributes
    assert e.children == {t: 1.0}

    # Verify we cannot add new attributes (slots behavior)
    with pytest.raises(AttributeError):
        x.new_attr = 1


def test_Expr_getitem():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    t1 = Term(x)
    t2 = Term(y)

    expr1 = Expr({t1: 2})
    assert expr1[t1] == 2
    assert expr1[x] == 2
    assert expr1[y] == 0
    assert expr1[t2] == 0

    expr2 = Expr({t1: 3, t2: 4})
    assert expr2[t1] == 3
    assert expr2[x] == 3
    assert expr2[t2] == 4
    assert expr2[y] == 4

    with pytest.raises(TypeError):
        expr2[1]

    expr3 = Expr({expr1: 1, expr2: 5})
    assert expr3[expr1] == 1
    assert expr3[expr2] == 5


def test_Expr_abs():
    m = Model()
    x = m.addVar("x")
    t = Term(x)
    expr = Expr({t: -3.0})
    abs_expr = abs(expr)

    assert isinstance(abs_expr, AbsExpr)
    assert str(abs_expr) == "AbsExpr(Expr({Term(x): -3.0}))"
    assert abs_expr._fchild() is expr


def test_expr_op_expr(model):
    m, x, y, z = model
    expr = x**1.5 + y
    assert isinstance(expr, Expr)
    expr += x**2
    assert isinstance(expr, Expr)
    expr += 1
    assert isinstance(expr, Expr)
    expr += x
    assert isinstance(expr, Expr)
    expr += 2 * y
    assert isinstance(expr, Expr)
    expr -= x**2
    assert isinstance(expr, Expr)
    expr -= 1
    assert isinstance(expr, Expr)
    expr -= x
    assert isinstance(expr, Expr)
    expr -= 2 * y
    assert isinstance(expr, Expr)
    expr *= x + y
    assert isinstance(expr, Expr)
    expr *= 2
    assert isinstance(expr, Expr)
    expr /= 2
    assert isinstance(expr, Expr)
    expr /= x + y
    assert isinstance(expr, Expr)
    assert isinstance(x**1.2 + x + y, Expr)
    assert isinstance(x**1.2 - x, Expr)
    assert isinstance(x**1.2 * (x + y), Expr)

    expr += x**2.2
    assert isinstance(expr, Expr)
    expr += sin(x)
    assert isinstance(expr, Expr)
    expr -= exp(x)
    assert isinstance(expr, Expr)
    expr /= log(x + 1)
    assert isinstance(expr, Expr)
    expr *= (x + y) ** 1.2
    assert isinstance(expr, Expr)
    expr /= exp(2)
    assert isinstance(expr, Expr)
    expr /= x + y
    assert isinstance(expr, Expr)
    expr = x**1.5 + y
    assert isinstance(expr, Expr)
    assert isinstance(sqrt(x) + expr, Expr)
    assert isinstance(exp(x) + expr, Expr)
    assert isinstance(sin(x) + expr, Expr)
    assert isinstance(cos(x) + expr, Expr)
    assert isinstance(1 / x + expr, Expr)
    assert isinstance(1 / x**1.5 - expr, Expr)
    assert isinstance(y / x - exp(expr), Expr)

    # sqrt(2) is not a constant expression and
    # we can only power to constant expressions!
    with pytest.raises(TypeError):
        expr **= sqrt(2)


# In contrast to Expr inequalities, we can't expect much of the sides
def test_inequality(model):
    m, x, y, z = model

    expr = x + 2 * y
    assert isinstance(expr, Expr)
    cons = expr <= x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, Expr)
    assert cons._lhs is None
    assert cons._rhs == 0.0

    assert isinstance(expr, Expr)
    cons = expr >= x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, Expr)
    assert cons._lhs == 0.0
    assert cons._rhs is None

    assert isinstance(expr, Expr)
    cons = expr >= 1 + x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, Expr)
    assert cons._lhs == 1
    assert cons._rhs is None

    assert isinstance(expr, Expr)
    cons = exp(expr) <= 1 + x**1.2
    assert isinstance(cons, ExprCons)
    assert isinstance(cons.expr, Expr)
    assert cons._rhs == 1
    assert cons._lhs is None


def test_equation(model):
    m, x, y, z = model
    equat = 2 * x**1.2 - 3 * sqrt(y) == 1
    assert isinstance(equat, ExprCons)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 1.0

    equat = exp(x + 2 * y) == 1 + x**1.2
    assert isinstance(equat, ExprCons)
    assert isinstance(equat.expr, Expr)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 1

    equat = x == 1 + x**1.2
    assert isinstance(equat, ExprCons)
    assert isinstance(equat.expr, Expr)
    assert equat._lhs == equat._rhs
    assert equat._lhs == 1


def test_rpow_constant_base(model):
    m, x, y, z = model
    a = 2**x
    b = exp(x * log(2.0))
    assert isinstance(a, Expr)
    assert repr(a) == repr(b)  # Structural equality is not implemented; compare strings
    m.addCons(2**x <= 1)

    with pytest.raises(ValueError):
        (-2) ** x
