from pyscipopt import Model
from pyscipopt.scip import (
    AbsExpr,
    ConstExpr,
    CosExpr,
    Expr,
    ProdExpr,
    SinExpr,
    SqrtExpr,
    Variable,
)


def test_init():
    m = Model()
    x = m.addVar("x")

    assert str(AbsExpr(x)) == "AbsExpr(Term(x))"
    assert str(SqrtExpr(10)) == "SqrtExpr(10.0)"
    assert (
        str(CosExpr(SinExpr(x) * x))
        == "CosExpr(ProdExpr({(SinExpr(Term(x)), Expr({Term(x): 1.0})): 1.0}))"
    )


def test_to_node():
    m = Model()
    x = m.addVar("x")

    expr = AbsExpr(x)
    assert expr._to_node() == [(Variable, x), (AbsExpr, 0)]
    assert expr._to_node(0) == []
    assert expr._to_node(10) == [
        (Variable, x),
        (AbsExpr, 0),
        (ConstExpr, 10),
        (ProdExpr, [1, 2]),
    ]


def test_neg():
    m = Model()
    x = m.addVar("x")

    expr = AbsExpr(x)
    res = -expr
    assert isinstance(res, Expr)
    assert str(res) == "Expr({AbsExpr(Term(x)): -1.0})"
