import pytest

from pyscipopt import Expr, Model, cos, exp, log, sin, sqrt
from pyscipopt.scip import (
    CONST,
    AbsExpr,
    ConstExpr,
    ExpExpr,
    PolynomialExpr,
    ProdExpr,
    Term,
    _ExprKey,
)


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    return m, x, y, z


def test_init_error(model):
    with pytest.raises(TypeError):
        Expr({42: 1})

    with pytest.raises(TypeError):
        Expr({"42": 0})

    m, x, y, z = model
    with pytest.raises(TypeError):
        Expr({x: 42})


def test_slots(model):
    m, x, y, z = model
    t = Term(x)
    e = Expr({t: 1.0})

    # Verify we can access defined slots/attributes
    assert e.children == {t: 1.0}

    # Verify we cannot add new attributes (slots behavior)
    with pytest.raises(AttributeError):
        x.new_attr = 1


def test_getitem(model):
    m, x, y, z = model
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


def test_abs():
    m = Model()
    x = m.addVar("x")
    t = Term(x)
    expr = Expr({t: -3.0})
    abs_expr = abs(expr)

    assert isinstance(abs_expr, AbsExpr)
    assert str(abs_expr) == "AbsExpr(Expr({Term(x): -3.0}))"


def test_fchild():
    m = Model()
    x = m.addVar("x")
    t = Term(x)

    expr1 = Expr({t: 1.0})
    assert expr1._fchild() == t

    expr2 = Expr({t: -1.0, expr1: 2.0})
    assert expr2._fchild() == t

    expr3 = Expr({expr1: 2.0, t: -1.0})
    assert expr3._fchild() == _ExprKey.wrap(expr1)


def test_add(model):
    m, x, y, z = model
    t = Term(x)

    expr1 = Expr({Term(x): 1.0}) + 1
    with pytest.raises(TypeError):
        expr1 + "invalid"

    with pytest.raises(TypeError):
        expr1 + []

    assert str(Expr() + Expr()) == "Expr({})"
    assert str(Expr() + 3) == "Expr({Term(): 3.0})"

    expr2 = Expr({t: 1.0})
    assert str(expr2 + 0) == "Expr({Term(x): 1.0})"
    assert str(expr2 + expr1) == "Expr({Term(x): 2.0, Term(): 1.0})"
    assert str(Expr({t: -1.0}) + expr1) == "Expr({Term(x): 0.0, Term(): 1.0})"
    assert (
        str(expr1 + cos(expr2))
        == "Expr({Term(x): 1.0, Term(): 1.0, CosExpr(Expr({Term(x): 1.0})): 1.0})"
    )
    assert (
        str(sqrt(expr2) + expr1)
        == "Expr({Term(x): 1.0, Term(): 1.0, SqrtExpr(Expr({Term(x): 1.0})): 1.0})"
    )
    assert (
        str(sqrt(expr2) + exp(expr1))
        == "Expr({SqrtExpr(Expr({Term(x): 1.0})): 1.0, ExpExpr(Expr({Term(x): 1.0, Term(): 1.0})): 1.0})"
    )


def test_iadd(model):
    m, x, y, z = model

    expr = log(x) + Expr({Term(x): 1.0})
    expr += 1
    assert str(expr) == "Expr({Term(x): 1.0, LogExpr(Term(x)): 1.0, Term(): 1.0})"

    expr += Expr({Term(x): 1.0})
    assert str(expr) == "Expr({Term(x): 2.0, LogExpr(Term(x)): 1.0, Term(): 1.0})"

    expr = x
    expr += sqrt(expr)
    assert str(expr) == "Expr({Term(x): 1.0, SqrtExpr(Term(x)): 1.0})"

    expr = sin(x)
    expr += cos(x)
    assert str(expr) == "Expr({SinExpr(Term(x)): 1.0, CosExpr(Term(x)): 1.0})"

    expr = exp(Expr({Term(x): 1.0}))
    expr += expr
    assert str(expr) == "Expr({ExpExpr(Expr({Term(x): 1.0})): 2.0})"


def test_mul(model):
    m, x, y, z = model
    expr1 = Expr({Term(x): 1.0, CONST: 1.0})

    with pytest.raises(TypeError):
        expr1 * "invalid"

    with pytest.raises(TypeError):
        expr1 * []

    assert str(Expr() * 3) == "Expr({Term(): 0.0})"

    expr2 = abs(expr1)
    assert (
        str(expr2 * expr2) == "PowExpr(AbsExpr(Expr({Term(x): 1.0, Term(): 1.0})), 2.0)"
    )

    assert str(Expr() * Expr()) == "Expr({Term(): 0.0})"
    assert str(expr1 * 0) == "Expr({Term(): 0.0})"
    assert str(expr1 * Expr()) == "Expr({Term(): 0.0})"
    assert str(Expr() * expr1) == "Expr({Term(): 0.0})"
    assert str(Expr({Term(x): 1.0, CONST: 0.0}) * 2) == "Expr({Term(x): 2.0})"
    assert (
        str(sin(expr1) * 2) == "Expr({SinExpr(Expr({Term(x): 1.0, Term(): 1.0})): 2.0})"
    )
    assert str(sin(expr1) * 1) == "SinExpr(Expr({Term(x): 1.0, Term(): 1.0}))"
    assert str(Expr({CONST: 2.0}) * expr1) == "Expr({Term(x): 2.0, Term(): 2.0})"


def test_imul(model):
    m, x, y, z = model

    expr = Expr({Term(x): 1.0, CONST: 1.0})
    expr *= 0
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = Expr({Term(x): 1.0, CONST: 1.0})
    expr *= 3
    assert str(expr) == "Expr({Term(x): 3.0, Term(): 3.0})"


def test_div(model):
    m, x, y, z = model

    expr1 = Expr({Term(x): 1.0, CONST: 1.0})
    with pytest.raises(ZeroDivisionError):
        expr1 / 0

    expr2 = expr1 / 2
    assert str(expr2) == "Expr({Term(x): 0.5, Term(): 0.5})"

    expr3 = 1 / x
    assert str(expr3) == "PowExpr(Expr({Term(x): 1.0}), -1.0)"

    expr4 = expr3 / expr3
    assert str(expr4) == "Expr({Term(): 1.0})"


def test_pow(model):
    m, x, y, z = model

    assert str((x + 2 * y) ** 0) == "Expr({Term(): 1.0})"

    with pytest.raises(TypeError):
        (x + y) ** "invalid"

    with pytest.raises(TypeError):
        x **= sqrt(2)


def test_rpow(model):
    m, x, y, z = model

    a = 2**x
    assert str(a) == (
        "ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(Expr({Term(): 2.0}))): 1.0}))"
    )

    b = exp(x * log(2.0))
    assert repr(a) == repr(b)  # Structural equality is not implemented; compare strings

    with pytest.raises(TypeError):
        "invalid" ** x

    with pytest.raises(ValueError):
        (-2) ** x


def test_sub(model):
    m, x, y, z = model

    expr1 = 2**x
    expr2 = exp(x * log(2.0))

    assert str(expr1 - expr2) == "Expr({Term(): 0.0})"
    assert str(expr2 - expr1) == "Expr({Term(): 0.0})"
    assert (
        str(expr1 - (expr2 + 1))
        == "Expr({Term(): -1.0, ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(Expr({Term(): 2.0}))): 1.0})): 0.0})"
    )
    assert (
        str(-expr2 + expr1)
        == "Expr({ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(Expr({Term(): 2.0}))): 1.0})): 0.0})"
    )
    assert (
        str(-expr1 - expr2)
        == "Expr({ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(Expr({Term(): 2.0}))): 1.0})): -2.0})"
    )


def test_isub(model):
    m, x, y, z = model

    expr = Expr({Term(x): 2.0, CONST: 3.0})
    expr -= 1
    assert str(expr) == "Expr({Term(x): 2.0, Term(): 2.0})"

    expr -= Expr({Term(x): 1.0})
    assert str(expr) == "Expr({Term(x): 1.0, Term(): 2.0})"

    expr = 2**x
    expr -= exp(x * log(2.0))
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = exp(x * log(2.0))
    expr -= 2**x
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = sin(x)
    expr -= cos(x)
    assert str(expr) == "Expr({CosExpr(Term(x)): -1.0, SinExpr(Term(x)): 1.0})"


def test_le(model):
    m, x, y, z = model

    expr1 = Expr({Term(x): 1.0})
    expr2 = Expr({CONST: 2.0})
    assert str(expr1 <= expr2) == "ExprCons(Expr({Term(x): 1.0}), None, 2.0)"
    assert str(expr2 <= expr1) == "ExprCons(Expr({Term(x): 1.0}), 2.0, None)"
    assert str(expr1 <= expr1) == "ExprCons(Expr({}), None, 0.0)"
    assert str(expr2 <= expr2) == "ExprCons(Expr({}), 0.0, None)"
    assert (
        str(sin(x) <= expr1)
        == "ExprCons(Expr({Term(x): -1.0, SinExpr(Term(x)): 1.0}), None, 0.0)"
    )

    expr3 = x + 2 * y
    expr4 = x**1.5
    assert (
        str(expr3 <= expr4)
        == "ExprCons(Expr({Term(x): 1.0, Term(y): 2.0, PowExpr(Expr({Term(x): 1.0}), 1.5): -1.0}), None, 0.0)"
    )
    assert (
        str(exp(expr3) <= 1 + expr4)
        == "ExprCons(Expr({PowExpr(Expr({Term(x): 1.0}), 1.5): -1.0, ExpExpr(Expr({Term(x): 1.0, Term(y): 2.0})): 1.0}), None, 1.0)"
    )

    with pytest.raises(TypeError):
        expr1 <= "invalid"


def test_ge(model):
    m, x, y, z = model

    expr1 = Expr({Term(x): 1.0, log(x): 2.0})
    expr2 = Expr({CONST: -1.0})
    assert (
        str(expr1 >= expr2)
        == "ExprCons(Expr({Term(x): 1.0, LogExpr(Term(x)): 2.0}), -1.0, None)"
    )
    assert (
        str(expr2 >= expr1)
        == "ExprCons(Expr({Term(x): 1.0, LogExpr(Term(x)): 2.0}), None, -1.0)"
    )
    assert str(expr1 >= expr1) == "ExprCons(Expr({}), 0.0, None)"
    assert str(expr2 >= expr2) == "ExprCons(Expr({}), None, 0.0)"

    expr3 = x + 2 * y
    expr4 = x**1.5
    assert (
        str(expr3 >= expr4)
        == "ExprCons(Expr({Term(x): 1.0, Term(y): 2.0, PowExpr(Expr({Term(x): 1.0}), 1.5): -1.0}), 0.0, None)"
    )
    assert (
        str(expr3 >= 1 + expr4)
        == "ExprCons(Expr({Term(x): 1.0, Term(y): 2.0, PowExpr(Expr({Term(x): 1.0}), 1.5): -1.0}), 1.0, None)"
    )

    with pytest.raises(TypeError):
        expr1 >= "invalid"


def test_eq(model):
    m, x, y, z = model

    expr1 = Expr({Term(x): -1.0, exp(x): 3.0})
    expr2 = Expr({expr1: -1.0})
    expr3 = Expr({CONST: 4.0})

    assert (
        str(expr2 == expr3)
        == "ExprCons(Expr({Expr({Term(x): -1.0, ExpExpr(Term(x)): 3.0}): -1.0}), 4.0, 4.0)"
    )
    assert (
        str(expr3 == expr2)
        == "ExprCons(Expr({Expr({Term(x): -1.0, ExpExpr(Term(x)): 3.0}): -1.0}), 4.0, 4.0)"
    )
    assert (
        str(2 * x**1.5 - 3 * sqrt(y) == 1)
        == "ExprCons(Expr({PowExpr(Expr({Term(x): 1.0}), 1.5): 2.0, SqrtExpr(Term(y)): -3.0}), 1.0, 1.0)"
    )
    assert (
        str(exp(x + 2 * y) == 1 + x**1.5)
        == "ExprCons(Expr({PowExpr(Expr({Term(x): 1.0}), 1.5): -1.0, ExpExpr(Expr({Term(x): 1.0, Term(y): 2.0})): 1.0}), 1.0, 1.0)"
    )
    assert (
        str(x == 1 + x**1.5)
        == "ExprCons(Expr({Term(x): 1.0, PowExpr(Expr({Term(x): 1.0}), 1.5): -1.0}), 1.0, 1.0)"
    )

    with pytest.raises(TypeError):
        expr1 == "invalid"


def test_to_dict(model):
    m, x, y, z = model

    expr = Expr({Term(x): 1.0, Term(y): -2.0, CONST: 3.0})

    children = expr._to_dict({})
    assert children == expr._children
    assert children is not expr._children
    assert len(children) == 3
    assert children[Term(x)] == 1.0
    assert children[Term(y)] == -2.0
    assert children[CONST] == 3.0

    children = expr._to_dict({Term(x): -1.0, sqrt(x): 0.0})
    assert children != expr._children
    assert len(children) == 4
    assert children[Term(x)] == 0.0
    assert children[Term(y)] == -2.0
    assert children[CONST] == 3.0
    assert children[_ExprKey.wrap(sqrt(x))] == 0.0

    children = expr._to_dict({Term(x): -1.0, Term(y): 2.0, CONST: -2.0}, copy=False)
    assert children is expr._children
    assert len(expr._children) == 3
    assert expr._children[Term(x)] == 0.0
    assert expr._children[Term(y)] == 0.0
    assert expr._children[CONST] == 1.0

    with pytest.raises(TypeError):
        expr._to_dict("invialid")


def test_normalize(model):
    m, x, y, z = model

    expr = Expr({Term(x): 2.0, Term(y): -4.0, CONST: 6.0})
    norm_expr = expr._normalize()
    assert expr is norm_expr
    assert str(norm_expr) == "Expr({Term(x): 2.0, Term(y): -4.0, Term(): 6.0})"

    expr = Expr({Term(x): 0.0, Term(y): 0.0, CONST: 0.0})
    norm_expr = expr._normalize()
    assert expr is norm_expr
    assert str(norm_expr) == "Expr({})"


def test_degree(model):
    m, x, y, z = model

    assert Expr({Term(x): 3.0, Term(y): -1.0}).degree() == 1
    assert Expr({Term(x, x): 2.0, Term(y): 4.0}).degree() == 2
    assert Expr({Term(x, y, z): 1.0, Term(y, y): -2.0}).degree() == 3
    assert Expr({CONST: 5.0}).degree() == 0
    assert Expr({CONST: 0.0, sin(x): 0.0}).degree() == float("inf")


def test_to_node(model):
    m, x, y, z = model

    expr = Expr({Term(x): 2.0, Term(y): -4.0, CONST: 6.0, sqrt(x): 0.0, exp(x): 1.0})

    assert expr._to_node(0) == []
    assert expr._to_node() == [
        (Term, x),
        (ConstExpr, 2.0),
        (ProdExpr, [0, 1]),
        (Term, y),
        (ConstExpr, -4.0),
        (ProdExpr, [3, 4]),
        (ConstExpr, 6.0),
        (Term, x),
        (ExpExpr, 7),
        (Expr, [2, 5, 6, 8]),
    ]
    assert expr._to_node(start=1) == [
        (Term, x),
        (ConstExpr, 2.0),
        (ProdExpr, [1, 2]),
        (Term, y),
        (ConstExpr, -4.0),
        (ProdExpr, [4, 5]),
        (ConstExpr, 6.0),
        (Term, x),
        (ExpExpr, 8),
        (Expr, [3, 6, 7, 9]),
    ]
    assert expr._to_node(coef=3, start=1) == [
        (Term, x),
        (ConstExpr, 2.0),
        (ProdExpr, [1, 2]),
        (Term, y),
        (ConstExpr, -4.0),
        (ProdExpr, [4, 5]),
        (ConstExpr, 6.0),
        (Term, x),
        (ExpExpr, 8),
        (Expr, [3, 6, 7, 9]),
        (ConstExpr, 3),
        (ProdExpr, [10, 11]),
    ]


def test_is_equal(model):
    m, x, y, z = model

    assert not Expr()._is_equal("invalid")
    assert Expr()._is_equal(Expr())
    assert Expr({CONST: 0.0, Term(x): 1.0})._is_equal(Expr({Term(x): 1.0, CONST: 0.0}))
    assert Expr({CONST: 0.0, Term(x): 1.0})._is_equal(
        PolynomialExpr({Term(x): 1.0, CONST: 0.0})
    )
    assert Expr({CONST: 0.0})._is_equal(PolynomialExpr({CONST: 0.0}))
    assert Expr({CONST: 0.0})._is_equal(ConstExpr(0.0))
