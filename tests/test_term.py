import pytest

from pyscipopt import Model
from pyscipopt.scip import ConstExpr, ProdExpr, Term


def test_init_error():
    with pytest.raises(TypeError):
        Term(1)

    x = Model().addVar("x")

    with pytest.raises(TypeError):
        Term(x, 1)

    with pytest.raises(TypeError):
        Term("invalid")


def test_slots():
    m = Model()
    x = m.addVar("x")
    t = Term(x)

    # Verify we can access defined slots/attributes
    assert t.vars == (x,)

    # Verify we cannot add new attributes (slots behavior)
    with pytest.raises(AttributeError):
        t.new_attr = 1


def test_mul():
    x = Model().addVar("x")
    t = Term(x)

    with pytest.raises(TypeError):
        "invalid" * t

    with pytest.raises(TypeError):
        t * 0

    with pytest.raises(TypeError):
        t * x

    assert t * t == Term(x, x)


def test_degree():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")

    t0 = Term()
    assert t0.degree() == 0

    t1 = Term(x)
    assert t1.degree() == 1

    t2 = Term(x, y)
    assert t2.degree() == 2

    t3 = Term(x, x, y)
    assert t3.degree() == 3


def test_to_node():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")

    t0 = Term()
    assert t0._to_node() == [(ConstExpr, 1)]
    assert t0._to_node(0) == []

    t1 = Term(x)
    assert t1._to_node() == [(Term, x)]
    assert t1._to_node(0) == []
    assert t1._to_node(-1) == [(Term, x), (ConstExpr, -1), (ProdExpr, [0, 1])]
    assert t1._to_node(-1, 2) == [(Term, x), (ConstExpr, -1), (ProdExpr, [2, 3])]

    t2 = Term(x, y)
    assert t2._to_node() == [(Term, x), (Term, y), (ProdExpr, [0, 1])]
    assert t2._to_node(3) == [
        (Term, x),
        (Term, y),
        (ConstExpr, 3),
        (ProdExpr, [0, 1, 2]),
    ]


def test_eq():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")

    t1 = Term(x)
    t2 = Term(y)

    assert t1 == Term(x)
    assert t1 != t2

    with pytest.raises(TypeError):
        t1 == x

    with pytest.raises(TypeError):
        t1 == 1


def test_getitem():
    x = Model().addVar("x")
    t = Term(x)

    assert x is t[0]

    with pytest.raises(TypeError):
        t[x]

    with pytest.raises(IndexError):
        t[1]

    with pytest.raises(IndexError):
        Term()[0]
