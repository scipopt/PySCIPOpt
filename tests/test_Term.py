import pytest

from pyscipopt import Model
from pyscipopt.scip import ConstExpr, ProdExpr, Term


@pytest.fixture(scope="module")
def model():
    m = Model()
    x = m.addVar("x")
    t = Term(x)
    return m, x, t


def test_init_error(model):
    with pytest.raises(TypeError):
        Term(1)

    m, x, t = model
    with pytest.raises(TypeError):
        Term(x, 1)

    with pytest.raises(TypeError):
        Term("invalid")


def test_slots(model):
    m, x, t = model

    # Verify we can access defined slots/attributes
    assert t.vars == (x,)

    # Verify we cannot add new attributes (slots behavior)
    with pytest.raises(AttributeError):
        t.new_attr = 1


def test_mul(model):
    m, x, t = model

    with pytest.raises(TypeError):
        "invalid" * t

    with pytest.raises(TypeError):
        t * 0

    with pytest.raises(TypeError):
        t * x

    t_square = t * t
    assert t_square == Term(x, x)
    assert str(t_square) == "Term(x, x)"


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
    assert t1 != x
    assert t1 != 1


def test_getitem(model):
    m, x, t = model

    assert x is t[0]

    with pytest.raises(TypeError):
        t[x]

    with pytest.raises(IndexError):
        t[1]

    with pytest.raises(IndexError):
        Term()[0]
