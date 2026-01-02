import pytest

from pyscipopt import Expr, Model, sin
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

    assert ProdExpr(Term(x), Term(y)).degree() == float("inf")


def test_add(model):
    m, x, y = model

    expr = ProdExpr(Term(x), Term(y))
    res = expr + sin(x)
    assert isinstance(res, Expr)
    assert (
        str(res)
        == "Expr({ProdExpr({(Term(x), Term(y)): 1.0}): 1.0, SinExpr(Term(x)): 1.0})"
    )

    expr += expr
    assert isinstance(expr, ProdExpr)
    assert str(expr) == "ProdExpr({(Term(x), Term(y)): 2.0})"


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
