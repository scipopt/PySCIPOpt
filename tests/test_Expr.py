import numpy as np
import pytest

from pyscipopt import Expr, Model, cos, exp, log, sin, sqrt
from pyscipopt.scip import (
    CONST,
    AbsExpr,
    ConstExpr,
    CosExpr,
    ExpExpr,
    LogExpr,
    PolynomialExpr,
    PowExpr,
    ProdExpr,
    SinExpr,
    SqrtExpr,
    Term,
    Variable,
    _ExprKey,
)


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    return m, x, y


def test_init_error(model):
    with pytest.raises(TypeError):
        Expr({42: 1})

    with pytest.raises(TypeError):
        Expr({"42": 0})

    with pytest.raises(TypeError):
        m, x, y = model
        Expr({x: 42})


def test_slots(model):
    m, x, y = model
    t = Term(x)
    e = Expr({t: 1.0})

    # Verify we can access defined slots/attributes
    assert e.children == {t: 1.0}

    # Verify we cannot add new attributes (slots behavior)
    with pytest.raises(AttributeError):
        x.new_attr = 1


def test_getitem(model):
    m, x, y = model
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


def test_add(model):
    m, x, y = model
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
        == "Expr({Term(x): 1.0, Term(): 1.0, CosExpr(Term(x)): 1.0})"
    )
    assert (
        str(sqrt(expr2) + expr1)
        == "Expr({Term(x): 1.0, Term(): 1.0, SqrtExpr(Term(x)): 1.0})"
    )

    expr3 = PolynomialExpr({t: 1.0, CONST: 1.0})
    assert (
        str(cos(expr2) + expr3)
        == "Expr({Term(x): 1.0, Term(): 1.0, CosExpr(Term(x)): 1.0})"
    )
    assert (
        str(sqrt(expr2) + exp(expr1))
        == "Expr({SqrtExpr(Term(x)): 1.0, ExpExpr(Expr({Term(x): 1.0, Term(): 1.0})): 1.0})"
    )

    assert (
        str(expr3 + exp(x * log(2.0)))
        == "Expr({Term(x): 1.0, Term(): 1.0, ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(2.0)): 1.0})): 1.0})"
    )

    # numpy array addition
    assert str(np.add(x, 2)) == "Expr({Term(x): 1.0, Term(): 2.0})"
    assert str(np.array([x]) + 2) == "[Expr({Term(x): 1.0, Term(): 2.0})]"
    assert str(1 + np.array([x])) == "[Expr({Term(x): 1.0, Term(): 1.0})]"
    assert (
        str(np.array([x, y]) + np.array([2]))
        == "[Expr({Term(x): 1.0, Term(): 2.0}) Expr({Term(y): 1.0, Term(): 2.0})]"
    )
    assert (
        str(np.array([[y]]) + np.array([[x]]))
        == "[[Expr({Term(y): 1.0, Term(x): 1.0})]]"
    )


def test_iadd(model):
    m, x, y = model

    expr = log(x) + Expr({Term(x): 1.0})
    expr += 1
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 1.0, LogExpr(Term(x)): 1.0, Term(): 1.0})"

    expr += Expr({Term(x): 1.0})
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 2.0, LogExpr(Term(x)): 1.0, Term(): 1.0})"

    expr = Expr({Term(x): 1.0})
    expr += PolynomialExpr({Term(x): 1.0})
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 2.0})"

    expr = PolynomialExpr({Term(x): 1.0})
    expr += PolynomialExpr({Term(x): 1.0})
    assert type(expr) is PolynomialExpr
    assert str(expr) == "Expr({Term(x): 2.0})"

    expr = Expr({Term(x): 1.0})
    expr += sqrt(expr)
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 1.0, SqrtExpr(Term(x)): 1.0})"

    expr = sin(x)
    expr += cos(x)
    assert type(expr) is Expr
    assert str(expr) == "Expr({SinExpr(Term(x)): 1.0, CosExpr(Term(x)): 1.0})"

    expr = exp(Expr({Term(x): 1.0}))
    expr += expr
    assert type(expr) is Expr
    assert str(expr) == "Expr({ExpExpr(Term(x)): 2.0})"


def test_mul(model):
    m, x, y = model
    expr = Expr({Term(x): 1.0, CONST: 1.0})

    with pytest.raises(TypeError):
        expr * "invalid"

    with pytest.raises(TypeError):
        expr * []

    assert str(Expr() * 3) == "Expr({Term(): 0.0})"

    expr2 = abs(expr)
    assert (
        str(expr2 * expr2) == "PowExpr(AbsExpr(Expr({Term(x): 1.0, Term(): 1.0})), 2.0)"
    )

    assert str(Expr() * Expr()) == "Expr({Term(): 0.0})"
    assert str(expr * 0) == "Expr({Term(): 0.0})"
    assert str(expr * Expr()) == "Expr({Term(): 0.0})"
    assert str(Expr() * expr) == "Expr({Term(): 0.0})"
    assert str(Expr({Term(x): 1.0, CONST: 0.0}) * 2) == "Expr({Term(x): 2.0})"
    assert (
        str(sin(expr) * 2) == "Expr({SinExpr(Expr({Term(x): 1.0, Term(): 1.0})): 2.0})"
    )
    assert str(sin(expr) * 1) == "SinExpr(Expr({Term(x): 1.0, Term(): 1.0}))"
    assert str(Expr({CONST: 2.0}) * expr) == "Expr({Term(x): 2.0, Term(): 2.0})"

    assert (
        str(Expr({Term(): -1.0}) * ProdExpr(Term(x), Term(y)))
        == "Expr({ProdExpr({(Term(x), Term(y)): 1.0}): -1.0})"
    )

    # numpy array multiplication
    assert str(np.multiply(x, 3)) == "Expr({Term(x): 3.0})"
    assert str(np.array([x]) * 3) == "[Expr({Term(x): 3.0})]"


def test_imul(model):
    m, x, y = model

    expr = Expr({Term(x): 1.0, CONST: 1.0})
    expr *= 0
    assert isinstance(expr, ConstExpr)
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = Expr({Term(x): 1.0, CONST: 1.0})
    expr *= 3
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 3.0, Term(): 3.0})"


def test_div(model):
    m, x, y = model

    expr1 = Expr({Term(x): 1.0, CONST: 1.0})
    with pytest.raises(ZeroDivisionError):
        expr1 / 0

    assert str(expr1 / 2) == "Expr({Term(x): 0.5, Term(): 0.5})"

    expr2 = 1 / x
    assert str(expr2) == "PowExpr(Expr({Term(x): 1.0}), -1.0)"

    assert str(expr2 / expr2) == "Expr({Term(): 1.0})"

    # test numpy array division
    assert str(np.divide(x, 2)) == "Expr({Term(x): 0.5})"
    assert str(np.array([x]) / 2) == "[Expr({Term(x): 0.5})]"


def test_pow(model):
    m, x, y = model

    assert str((x + 2 * y) ** 0) == "Expr({Term(): 1.0})"

    with pytest.raises(TypeError):
        (x + y) ** "invalid"

    with pytest.raises(TypeError):
        x **= sqrt(2)

    # test numpy array power
    assert str(np.power(x, 3)) == "Expr({Term(x, x, x): 1.0})"
    assert str(np.array([x]) ** 3) == "[Expr({Term(x, x, x): 1.0})]"


def test_rpow(model):
    m, x, y = model

    expr1 = 2**x
    assert str(expr1) == (
        "ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(2.0)): 1.0}))"
    )

    expr2 = exp(x * log(2.0))
    # Structural equality is not implemented; compare strings
    assert repr(expr1) == repr(expr2)

    with pytest.raises(TypeError):
        "invalid" ** x

    with pytest.raises(ValueError):
        (-2) ** x


def test_sub(model):
    m, x, y = model

    expr1 = 2**x
    expr2 = exp(x * log(2.0))

    assert str(expr1 - expr2) == "Expr({Term(): 0.0})"
    assert str(expr2 - expr1) == "Expr({Term(): 0.0})"
    assert (
        str(expr1 - (expr2 + 1))
        == "Expr({Term(): -1.0, ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(2.0)): 1.0})): 0.0})"
    )
    assert (
        str(-expr2 + expr1)
        == "Expr({ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(2.0)): 1.0})): 0.0})"
    )
    assert (
        str(-expr1 - expr2)
        == "Expr({ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(2.0)): 1.0})): -2.0})"
    )

    assert (
        str(1 - expr1)
        == "Expr({ExpExpr(ProdExpr({(Expr({Term(x): 1.0}), LogExpr(2.0)): 1.0})): -1.0, Term(): 1.0})"
    )

    # test numpy array subtraction
    assert str(np.subtract(x, 2)) == "Expr({Term(x): 1.0, Term(): -2.0})"
    assert str(np.array([x]) - 2) == "[Expr({Term(x): 1.0, Term(): -2.0})]"


def test_isub(model):
    m, x, y = model

    expr = Expr({Term(x): 2.0, CONST: 3.0})
    expr -= 1
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 2.0, Term(): 2.0})"

    expr -= Expr({Term(x): 1.0})
    assert type(expr) is Expr
    assert str(expr) == "Expr({Term(x): 1.0, Term(): 2.0})"

    expr = 2**x
    expr -= exp(x * log(2.0))
    assert isinstance(expr, ConstExpr)
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = exp(x * log(2.0))
    expr -= 2**x
    assert isinstance(expr, ConstExpr)
    assert str(expr) == "Expr({Term(): 0.0})"

    expr = sin(x)
    expr -= cos(x)
    assert type(expr) is Expr
    assert str(expr) == "Expr({CosExpr(Term(x)): -1.0, SinExpr(Term(x)): 1.0})"


def test_le(model):
    m, x, y = model

    expr1 = Expr({Term(x): 1.0})
    expr2 = Expr({CONST: 2.0})
    assert str(expr1 <= expr2) == "ExprCons(Expr({Term(x): 1.0}), None, 2.0)"
    assert str(expr2 <= expr1) == "ExprCons(Expr({Term(x): -1.0}), None, -2.0)"
    assert str(expr1 <= expr1) == "ExprCons(Expr({}), None, 0.0)"
    assert str(expr2 <= expr2) == "ExprCons(Expr({}), None, 0.0)"
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

    # test numpy array less equal
    assert str(np.less_equal(x, 2)) == "ExprCons(Expr({Term(x): 1.0}), None, 2.0)"

    with pytest.raises(TypeError):
        expr1 <= "invalid"

    with pytest.raises(TypeError):
        1 <= expr1 <= 1


def test_ge(model):
    m, x, y = model

    expr1 = Expr({Term(x): 1.0, log(x): 2.0})
    expr2 = Expr({CONST: -1.0})
    assert (
        str(expr1 >= expr2)
        == "ExprCons(Expr({Term(x): 1.0, LogExpr(Term(x)): 2.0}), -1.0, None)"
    )
    assert (
        str(expr2 >= expr1)
        == "ExprCons(Expr({Term(x): -1.0, LogExpr(Term(x)): -2.0}), 1.0, None)"
    )
    assert str(expr1 >= expr1) == "ExprCons(Expr({}), 0.0, None)"
    assert str(expr2 >= expr2) == "ExprCons(Expr({}), 0.0, None)"

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

    # test numpy array greater equal
    assert str(np.greater_equal(x, 2)) == "ExprCons(Expr({Term(x): 1.0}), 2.0, None)"

    with pytest.raises(TypeError):
        expr1 >= "invalid"


def test_eq(model):
    m, x, y = model

    expr1 = Expr({Term(x): -1.0, exp(x): 3.0})
    expr2 = Expr({expr1: -1.0})
    expr3 = Expr({CONST: 4.0})

    assert (
        str(expr2 == expr3)
        == "ExprCons(Expr({Expr({Term(x): -1.0, ExpExpr(Term(x)): 3.0}): -1.0}), 4.0, 4.0)"
    )
    assert (
        str(expr3 == expr2)
        == "ExprCons(Expr({Expr({Term(x): -1.0, ExpExpr(Term(x)): 3.0}): 1.0}), -4.0, -4.0)"
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

    # test numpy array equal
    assert str(np.equal(x, 2)) == "ExprCons(Expr({Term(x): 1.0}), 2.0, 2.0)"

    with pytest.raises(TypeError):
        expr1 == "invalid"


def test_normalize(model):
    m, x, y = model

    expr = Expr({Term(x): 2.0, Term(y): -4.0, CONST: 6.0})
    norm_expr = expr._normalize()
    assert expr is norm_expr
    assert str(norm_expr) == "Expr({Term(x): 2.0, Term(y): -4.0, Term(): 6.0})"

    expr = Expr({Term(x): 0.0, Term(y): 0.0, CONST: 0.0})
    norm_expr = expr._normalize()
    assert expr is norm_expr
    assert str(norm_expr) == "Expr({})"


def test_degree(model):
    m, x, y = model
    z = m.addVar("z")

    assert Expr({Term(x): 3.0, Term(y): -1.0}).degree() == 1
    assert Expr({Term(x, x): 2.0, Term(y): 4.0}).degree() == 2
    assert Expr({Term(x, y, z): 1.0, Term(y, y): -2.0}).degree() == 3
    assert Expr({CONST: 5.0}).degree() == 0
    assert Expr({CONST: 0.0, sin(x): 0.0}).degree() == float("inf")


def test_to_node(model):
    m, x, y = model

    expr = Expr(
        {
            Term(x): 2.0,
            Term(y): -4.0,
            CONST: 6.0,
            _ExprKey(sqrt(x)): 0.0,
            _ExprKey(exp(x)): 1.0,
        }
    )

    assert expr._to_node(0) == []
    assert expr._to_node() == [
        (Variable, x),
        (ConstExpr, 2.0),
        (ProdExpr, [0, 1]),
        (Variable, y),
        (ConstExpr, -4.0),
        (ProdExpr, [3, 4]),
        (ConstExpr, 6.0),
        (Variable, x),
        (ExpExpr, 7),
        (Expr, [2, 5, 6, 8]),
    ]
    assert expr._to_node(start=1) == [
        (Variable, x),
        (ConstExpr, 2.0),
        (ProdExpr, [1, 2]),
        (Variable, y),
        (ConstExpr, -4.0),
        (ProdExpr, [4, 5]),
        (ConstExpr, 6.0),
        (Variable, x),
        (ExpExpr, 8),
        (Expr, [3, 6, 7, 9]),
    ]
    assert expr._to_node(coef=3, start=1) == [
        (Variable, x),
        (ConstExpr, 6.0),
        (ProdExpr, [1, 2]),
        (Variable, y),
        (ConstExpr, -12.0),
        (ProdExpr, [4, 5]),
        (ConstExpr, 18.0),
        (Variable, x),
        (ExpExpr, 8),
        (ConstExpr, 3.0),
        (ProdExpr, [9, 10]),
        (Expr, [3, 6, 7, 11]),
    ]


def test_is_equal(model):
    m, x, y = model

    assert _ExprKey(Expr()) != "invalid"
    assert _ExprKey(Expr()) == _ExprKey(Expr())
    assert _ExprKey(Expr({CONST: 0.0, Term(x): 1.0})) == _ExprKey(
        Expr({Term(x): 1.0, CONST: 0.0})
    )
    assert _ExprKey(Expr({CONST: 0.0, Term(x): 1.0})) == _ExprKey(
        PolynomialExpr({Term(x): 1.0, CONST: 0.0})
    )
    assert _ExprKey(Expr({CONST: 0.0})) == _ExprKey(PolynomialExpr({CONST: 0.0}))
    assert _ExprKey(Expr({CONST: 0.0})) == _ExprKey(ConstExpr(0.0))

    assert _ExprKey(ProdExpr(Term(x), Term(y))) != _ExprKey(
        PowExpr(ProdExpr(Term(x), Term(y)), 1.0)
    )
    assert _ExprKey(ProdExpr(Term(x), Term(y))) == _ExprKey(
        ProdExpr(Term(x), Term(y)) * 1.0
    )

    assert _ExprKey(PowExpr(Term(x), -1.0)) != _ExprKey(PowExpr(Term(x), 1.0))
    assert _ExprKey(PowExpr(Term(x), 1)) == _ExprKey(PowExpr(Term(x), 1.0))

    assert _ExprKey(CosExpr(Term(x))) != _ExprKey(SinExpr(Term(x)))
    assert _ExprKey(LogExpr(Term(x))) == _ExprKey(LogExpr(Term(x)))


def test_neg(model):
    m, x, y = model

    expr1 = -Expr({Term(x): 1.0, CONST: -2.0})
    assert type(expr1) is Expr
    assert str(expr1) == "Expr({Term(x): -1.0, Term(): 2.0})"

    expr2 = -(sin(x) + cos(y))
    assert type(expr2) is Expr
    assert str(expr2) == "Expr({SinExpr(Term(x)): -1.0, CosExpr(Term(y)): -1.0})"

    # test numpy array negation
    assert str(np.negative(x)) == "Expr({Term(x): -1.0})"
    assert (
        str(np.negative(np.array([x, y])))
        == "[Expr({Term(x): -1.0}) Expr({Term(y): -1.0})]"
    )


def test_sin(model):
    m, x, y = model

    expr1 = sin(1)
    assert isinstance(expr1, SinExpr)
    assert str(expr1) == "SinExpr(1.0)"
    assert str(ConstExpr(1.0).sin()) == str(expr1)
    assert str(SinExpr(1.0)) == str(expr1)
    assert str(sin(ConstExpr(1.0))) == str(expr1)

    expr2 = Expr({Term(x): 1.0})
    expr3 = Expr({Term(x, y): 1.0})
    assert isinstance(sin(expr2), SinExpr)
    assert isinstance(sin(expr3), SinExpr)

    array = [expr2, expr3]
    assert type(sin(array)) is np.ndarray
    assert str(sin(array)) == "[SinExpr(Term(x)) SinExpr(Term(x, y))]"
    assert str(np.sin(array)) == str(sin(array))
    assert str(sin(np.array(array))) == str(sin(array))
    assert str(np.sin(np.array(array))) == str(sin(array))


def test_cos(model):
    m, x, y = model

    expr1 = Expr({Term(x): 1.0})
    expr2 = Expr({Term(x, y): 1.0})
    assert isinstance(cos(expr1), CosExpr)
    assert str(cos([expr1, expr2])) == "[CosExpr(Term(x)) CosExpr(Term(x, y))]"


def test_exp(model):
    m, x, y = model

    expr = Expr({ProdExpr(Term(x), Term(y)): 1.0})
    assert isinstance(exp(expr), ExpExpr)
    assert str(exp(expr)) == "ExpExpr(Expr({ProdExpr({(Term(x), Term(y)): 1.0}): 1.0}))"
    assert str(expr.exp()) == str(exp(expr))


def test_log(model):
    m, x, y = model

    expr = AbsExpr(Expr({Term(x): 1.0}) + Expr({Term(y): 1.0}))
    assert isinstance(log(expr), LogExpr)
    assert str(log(expr)) == "LogExpr(AbsExpr(Expr({Term(x): 1.0, Term(y): 1.0})))"
    assert str(expr.log()) == str(log(expr))


def test_sqrt(model):
    m, x, y = model

    expr = Expr({Term(x): 2.0})
    assert isinstance(sqrt(expr), SqrtExpr)
    assert str(sqrt(expr)) == "SqrtExpr(Expr({Term(x): 2.0}))"
    assert str(expr.sqrt()) == str(sqrt(expr))


def test_abs(model):
    m, x, y = model

    expr = Expr({Term(x): -3.0})
    assert isinstance(abs(expr), AbsExpr)
    assert str(abs(expr)) == "AbsExpr(Expr({Term(x): -3.0}))"
    assert str(np.abs(Expr({Term(x): -3.0}))) == str(abs(expr))


def test_cmp(model):
    m, x, y = model

    with pytest.raises(NotImplementedError):
        Expr({Term(x): -3.0}) > y

    with pytest.raises(NotImplementedError):
        Expr({Term(x): -3.0}) < y


def test_array_ufunc(model):
    m, x, y = model

    with pytest.raises(TypeError):
        np.floor_divide(x, 2)

    assert x.__array_ufunc__(None, "invalid") == NotImplemented
