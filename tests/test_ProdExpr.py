import pytest

from pyscipopt import Expr, Model, exp, log, sin, sqrt
from pyscipopt.scip import PolynomialExpr, PowExpr, ProdExpr, Term


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
