import pytest

from pyscipopt import Expr, Model, Variable, sin, sqrt
from pyscipopt.scip import CONST, AbsExpr, ConstExpr, PolynomialExpr, ProdExpr, Term


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


def test_add(model):
    m, x, y = model

    expr = PolynomialExpr({Term(x): 2.0, Term(y): 4.0}) + 3
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 2.0, Term(y): 4.0, Term(): 3.0})"

    expr = PolynomialExpr({Term(x): 2.0}) + (-2 * x)
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 0.0})"

    expr = PolynomialExpr() + 0
    assert type(expr) is ConstExpr
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = PolynomialExpr() + 1
    assert type(expr) is ConstExpr
    assert str(expr) == "Expr({Term(): 1.0})"


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


def test_mul(model):
    m, x, y = model

    expr = PolynomialExpr({Term(x): 2.0, Term(y): 4.0}) * 3
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 6.0, Term(y): 12.0})"

    expr = PolynomialExpr({Term(x): 2.0}) * PolynomialExpr({Term(x): 1.0, Term(y): 1.0})
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x, x): 2.0, Term(x, y): 2.0})"

    expr = ConstExpr(1.0) * PolynomialExpr()
    assert type(expr) is ConstExpr
    assert str(expr) == "Expr({Term(): 0.0})"


def test_div(model):
    m, x, y = model

    expr = PolynomialExpr({Term(x): 2.0, Term(y): 4.0}) / 2
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 1.0, Term(y): 2.0})"

    expr = PolynomialExpr({Term(x): 2.0}) / x
    assert type(expr) is ProdExpr
    assert (
        str(expr)
        == "ProdExpr({(Expr({Term(x): 2.0}), PowExpr(Expr({Term(x): 1.0}), -1.0)): 1.0})"
    )


def test_to_node(model):
    m, x, y = model

    expr = PolynomialExpr()
    assert expr._to_node() == []
    assert expr._to_node(2) == []

    expr = ConstExpr(0.0)
    assert expr._to_node() == []
    assert expr._to_node(3) == []

    expr = ConstExpr(-1)
    assert expr._to_node() == [(ConstExpr, -1.0)]
    assert expr._to_node(2) == [(ConstExpr, -1.0), (ConstExpr, 2.0), (ProdExpr, [0, 1])]

    expr = PolynomialExpr({Term(x): 2.0, Term(y): 4.0})
    assert expr._to_node() == [
        (Variable, x),
        (ConstExpr, 2.0),
        (ProdExpr, [0, 1]),
        (Variable, y),
        (ConstExpr, 4.0),
        (ProdExpr, [3, 4]),
        (Expr, [2, 5]),
    ]


def test_abs(model):
    m, x, y = model

    expr = abs(PolynomialExpr({Term(x): -2.0, Term(y): 4.0}))
    assert isinstance(expr, AbsExpr)
    assert str(expr) == "AbsExpr(Expr({Term(x): -2.0, Term(y): 4.0}))"

    expr = abs(ConstExpr(-3.0))
    assert isinstance(expr, ConstExpr)
    assert str(expr) == "Expr({Term(): 3.0})"


def test_neg(model):
    m, x, y = model

    expr = -PolynomialExpr({Term(x): -2.0, Term(y): 4.0})
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 2.0, Term(y): -4.0})"

    expr = -ConstExpr(-3.0)
    assert isinstance(expr, ConstExpr)
    assert str(expr) == "Expr({Term(): 3.0})"
