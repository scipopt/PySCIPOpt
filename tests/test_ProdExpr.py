import pytest

from pyscipopt import Expr, Model, exp, sin, sqrt
from pyscipopt.scip import (
    ConstExpr,
    PolynomialExpr,
    PowExpr,
    ProdExpr,
    SinExpr,
    Term,
    Variable,
)


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    return m, x, y


def test_init(model):
    m, x, y = model

    with pytest.raises(ValueError):
        ProdExpr(Term(x), Term(x))


def test_degree(model):
    m, x, y = model

    expr = exp(x) * y
    assert expr.degree() == float("inf")


def test_add(model):
    m, x, y = model

    expr = sqrt(x) * y
    res = expr + sin(x)
    assert type(res) is Expr
    assert (
        str(res)
        == "Expr({ProdExpr({(SqrtExpr(Term(x)), Expr({Term(y): 1.0})): 1.0}): 1.0, SinExpr(Term(x)): 1.0})"
    )

    res = expr + expr
    assert isinstance(expr, ProdExpr)
    assert str(res) == "ProdExpr({(SqrtExpr(Term(x)), Expr({Term(y): 1.0})): 2.0})"

    expr = sqrt(x) * y
    expr += expr
    assert isinstance(expr, ProdExpr)
    assert str(expr) == "ProdExpr({(SqrtExpr(Term(x)), Expr({Term(y): 1.0})): 2.0})"

    expr += 1
    assert type(expr) is Expr
    assert (
        str(expr)
        == "Expr({Term(): 1.0, ProdExpr({(SqrtExpr(Term(x)), Expr({Term(y): 1.0})): 2.0}): 1.0})"
    )


def test_mul(model):
    m, x, y = model

    expr = ProdExpr(Term(x), Term(y))
    res = expr * 3
    assert isinstance(res, ProdExpr)
    assert str(res) == "ProdExpr({(Term(x), Term(y)): 3.0})"

    expr *= 3
    assert isinstance(res, ProdExpr)
    assert str(res) == "ProdExpr({(Term(x), Term(y)): 3.0})"

    expr *= expr
    assert isinstance(expr, PowExpr)
    assert str(expr) == "PowExpr(ProdExpr({(Term(x), Term(y)): 3.0}), 2.0)"


def test_div(model):
    m, x, y = model

    expr = 2 * (sin(x) * y)
    assert (
        str(expr / 0.5) == "ProdExpr({(SinExpr(Term(x)), Expr({Term(y): 1.0})): 4.0})"
    )
    assert (
        str(expr / x)
        == "ProdExpr({(ProdExpr({(SinExpr(Term(x)), Expr({Term(y): 1.0})): 2.0}), PowExpr(Expr({Term(x): 1.0}), -1.0)): 1.0})"
    )
    assert str(expr / expr) == "Expr({Term(): 1.0})"


def test_cmp(model):
    m, x, y = model

    expr1 = sin(x) * y
    expr2 = y * sin(x)
    assert str(expr1 == expr2) == "ExprCons(Expr({}), 0.0, 0.0)"
    assert (
        str(expr1 == 1)
        == "ExprCons(ProdExpr({(SinExpr(Term(x)), Expr({Term(y): 1.0})): 1.0}), 1.0, 1.0)"
    )


def test_normalize(model):
    m, x, y = model

    expr = ProdExpr()._normalize()
    assert isinstance(expr, ConstExpr)
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = sin(x) * y
    assert isinstance(expr, ProdExpr)
    assert str(expr - expr) == "Expr({Term(): 0.0})"

    expr = ProdExpr(Term(x))._normalize()
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 1.0})"

    expr = ProdExpr(sin(x))._normalize()
    assert isinstance(expr, SinExpr)
    assert str(expr) == "SinExpr(Term(x))"

    expr = sin(x) * y
    assert str(expr._normalize()) == str(expr)


def test_to_node(model):
    m, x, y = model

    expr = ProdExpr()
    assert expr._to_node() == []
    assert expr._to_node(0) == []
    assert expr._to_node(10) == []

    expr = ProdExpr(Term(x), Term(y))
    assert expr._to_node() == [(Variable, x), (Variable, y), (ProdExpr, [0, 1])]
    assert expr._to_node(0) == []
    assert (expr * 2)._to_node() == [
        (Variable, x),
        (Variable, y),
        (ConstExpr, 2),
        (ProdExpr, [0, 1, 2]),
    ]
