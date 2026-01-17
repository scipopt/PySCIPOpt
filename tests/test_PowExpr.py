import pytest

from pyscipopt import Model
from pyscipopt.scip import ConstExpr, PolynomialExpr, PowExpr, ProdExpr, Term, Variable


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    return m, x, y


def test_degree(model):
    m, x, y = model

    assert PowExpr(Term(x), 3.0).degree() == float("inf")


def test_mul(model):
    m, x, y = model

    expr = PowExpr(Term(x), 2.0)
    res = expr * expr
    assert isinstance(res, PowExpr)
    assert str(res) == "PowExpr(Term(x), 4.0)"

    res = expr * PowExpr(Term(x), 1.0)
    assert isinstance(res, PowExpr)
    assert str(res) == "PowExpr(Term(x), 3.0)"

    res = expr * PowExpr(Term(x), -1.0)
    assert isinstance(res, PolynomialExpr)
    assert str(res) == "Expr({Term(x): 1.0})"

    res = PowExpr(Term(x), 1.0) * PowExpr(Term(x), -1.0)
    assert isinstance(res, ConstExpr)
    assert str(res) == "Expr({Term(): 1.0})"

    res = PowExpr(Term(x), 1.0) * PowExpr(Term(x), -1.0)
    assert isinstance(res, ConstExpr)
    assert str(res) == "Expr({Term(): 1.0})"


def test_imul(model):
    m, x, y = model

    expr = PowExpr(Term(x), 2.0)
    expr *= expr
    assert isinstance(expr, PowExpr)
    assert str(expr) == "PowExpr(Term(x), 4.0)"

    expr = PowExpr(Term(x), 2.0)
    expr *= PowExpr(Term(x), 1.0)
    assert isinstance(expr, PowExpr)
    assert str(expr) == "PowExpr(Term(x), 3.0)"

    expr = PowExpr(Term(x), 2.0)
    expr *= PowExpr(Term(x), -1.0)
    assert isinstance(expr, PolynomialExpr)
    assert str(expr) == "Expr({Term(x): 1.0})"

    expr = PowExpr(Term(x), 1.0)
    expr *= PowExpr(Term(x), -1.0)
    assert isinstance(expr, ConstExpr)
    assert str(expr) == "Expr({Term(): 1.0})"

    expr = PowExpr(Term(x), 1.0)
    expr *= x
    assert isinstance(expr, ProdExpr)
    assert str(expr) == "ProdExpr({(PowExpr(Term(x), 1.0), Expr({Term(x): 1.0})): 1.0})"


def test_div(model):
    m, x, y = model

    expr = PowExpr(Term(x), 2.0)
    res = expr / PowExpr(Term(x), 1.0)
    assert isinstance(res, PolynomialExpr)
    assert str(res) == "Expr({Term(x): 1.0})"

    expr = PowExpr(Term(x), 2.0)
    res = expr / expr
    assert isinstance(res, ConstExpr)
    assert str(res) == "Expr({Term(): 1.0})"

    expr = PowExpr(Term(x), 2.0)
    res = expr / x
    assert isinstance(res, ProdExpr)
    assert (
        str(res)
        == "ProdExpr({(PowExpr(Term(x), 2.0), PowExpr(Expr({Term(x): 1.0}), -1.0)): 1.0})"
    )


def test_cmp(model):
    m, x, y = model

    expr1 = PowExpr(Term(x), 2.0)
    expr2 = PowExpr(Term(y), -2.0)

    assert (
        str(expr1 == expr2)
        == "ExprCons(Expr({PowExpr(Term(y), -2.0): -1.0, PowExpr(Term(x), 2.0): 1.0}), 0.0, 0.0)"
    )
    assert str(expr1 <= 1) == "ExprCons(PowExpr(Term(x), 2.0), None, 1.0)"


def test_normalize(model):
    m, x, y = model

    assert str(PowExpr(Term(x), 2.0)._normalize()) == "PowExpr(Term(x), 2.0)"
    assert str(PowExpr(Term(x), 1.0)._normalize()) == "Expr({Term(x): 1.0})"
    assert str(PowExpr(Term(x), 0.0)._normalize()) == "Expr({Term(): 1.0})"


def test_to_node(model):
    m, x, y = model

    expr = PowExpr(Term(x), 1)
    assert expr._to_node() == [(Variable, x), (ConstExpr, 1), (PowExpr, [0, 1])]
    assert expr._to_node(0) == []
    assert expr._to_node(10) == [
        (Variable, x),
        (ConstExpr, 1),
        (PowExpr, [0, 1]),
        (ConstExpr, 10),
        (ProdExpr, [2, 3]),
    ]

    expr = PowExpr(ProdExpr(Term(x), Term(y)), -1)
    assert expr._to_node() == [
        (Variable, x),
        (Variable, y),
        (ProdExpr, [0, 1]),
        (ConstExpr, -1.0),
        (PowExpr, [2, 3]),
    ]
