import pytest

from pyscipopt import Expr, Model, sin, sqrt
from pyscipopt.scip import CONST, ConstExpr, PolynomialExpr, Term


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    return m, x, y


def test_init_error(model):
    m, x, y = model

    with pytest.raises(TypeError):
        PolynomialExpr({x: 1.0})

    with pytest.raises(TypeError):
        PolynomialExpr({Expr({Term(x): 1.0}): 1.0})

    with pytest.raises(TypeError):
        ConstExpr("invalid")


def test_iadd(model):
    m, x, y = model

    expr = ConstExpr(2.0)
    expr += 0
    assert type(expr) is ConstExpr
    assert str(expr) == "Expr({Term(): 2.0})"

    expr = ConstExpr(2.0)
    expr += Expr({CONST: 0.0})
    assert type(expr) is ConstExpr
    assert str(expr) == "Expr({Term(): 2.0})"

    expr = ConstExpr(2.0)
    expr += Expr()
    assert type(expr) is ConstExpr
    assert str(expr) == "Expr({Term(): 2.0})"

    expr = ConstExpr(2.0)
    expr += -2
    assert type(expr) is ConstExpr
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = ConstExpr(2.0)
    expr += sin(x)
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(): 2.0, SinExpr(Term(x)): 1.0})"

    expr = x
    expr += -x
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 0.0})"

    expr = x
    expr += 0
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 1.0})"

    expr = x
    expr += PolynomialExpr({Term(x): 1.0, Term(y): 1.0})
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 2.0, Term(y): 1.0})"

    expr = PolynomialExpr({Term(x): 1.0, Term(): 1.0})
    expr += -x
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 0.0, Term(): 1.0})"

    expr = PolynomialExpr({Term(x): 1.0, Term(y): 1.0})
    expr += sqrt(x)
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 1.0, Term(y): 1.0, SqrtExpr(Term(x)): 1.0})"

    expr = PolynomialExpr({Term(x): 1.0, Term(y): 1.0})
    expr += Expr({CONST: 0.0})
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 1.0, Term(y): 1.0})"

    expr = PolynomialExpr({Term(x): 1.0, Term(y): 1.0})
    expr += sqrt(x)
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 1.0, Term(y): 1.0, SqrtExpr(Term(x)): 1.0})"
