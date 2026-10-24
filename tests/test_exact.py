from fractions import Fraction

import pytest

from pyscipopt import Model, ExactModel, ExactVariable, Variable


def _exact_supported():
    try:
        ExactModel()
    except ValueError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _exact_supported(), reason="SCIP was built without exact solving support")


def test_exact_model():
    m = ExactModel()
    assert isinstance(m, Model)
    assert not hasattr(Model(), "chgVarObj")
    assert not hasattr(m, "addVarExact")
    with pytest.raises(TypeError):
        Model(exact=True)
    with pytest.raises(TypeError):
        ExactModel(sourceModel=m)
    assert isinstance(Model.from_ptr(m.to_ptr(False), False), ExactModel)

    x = m.addVar("x", lb="1/3", ub=Fraction(10, 3), obj=-1)
    assert isinstance(x, ExactVariable)
    assert all(isinstance(v, ExactVariable) for v in m.getVars())
    assert x.getLbOriginal() == Fraction(1, 3)
    assert x.getLbGlobal() == Fraction(1, 3)
    assert x.getUbLocal() == Fraction(10, 3)
    assert x.getUbOriginal() == Fraction(10, 3)
    assert x.getObj() == -1
    y = m.addVar("y", lb=None, ub=float("inf"))
    assert y.getLbOriginal() == -float("inf")
    assert y.getUbOriginal() == float("inf")
    b = m.addVar("b", vtype="B")
    assert b.getUbOriginal() == 1
    assert b.vtype() == "BINARY"

    m.chgVarObj(x, Fraction(-1, 7))
    m.chgVarLb(x, None)
    m.chgVarUb(x, "5/2")
    assert x.getObj() == Fraction(-1, 7)
    assert x.getLbOriginal() == -float("inf")
    assert x.getUbOriginal() == Fraction(5, 2)

    with pytest.raises(ValueError):
        m.addVar("bad", lb=2, ub=1)
    with pytest.raises(Warning):
        m.addVar("bad", vtype="Q")
    with pytest.raises(ValueError):
        m.addVar("bad", lb="abc")
    with pytest.raises(ValueError):
        m.addVar("bad", obj=float("inf"))
    with pytest.raises(ValueError):
        m.addVar("bad", pricedVar=True)
    with pytest.raises(TypeError):
        m.setObjective(x)
    with pytest.raises(TypeError):
        m.setObjective(Fraction(1, 3) * x)

    n = Model()
    z = n.addVar("z", lb=0.5)
    assert isinstance(z.getLbOriginal(), float)
    assert type(z) is Variable
    assert not hasattr(z, "getLbOriginalExact")
    assert not hasattr(x, "getLbOriginalExact")


RAISING = ["tightenVarLbGlobal", "tightenVarUbGlobal", "chgVarObjDive", "chgVarLbProbing",
           "chgVarUbProbing", "chgVarObjProbing", "fixVarProbing"]


def test_bound_changes():
    m = ExactModel()
    x = m.addVar("x", lb=0, ub=10)
    m.chgVarLbGlobal(x, "1/3")
    m.chgVarUbGlobal(x, Fraction(7, 2))
    assert x.getLbGlobal() == Fraction(1, 3)
    assert x.getUbGlobal() == Fraction(7, 2)
    assert m.tightenVarLb(x, Fraction(1, 2)) == (False, True)
    assert x.getLbLocal() == Fraction(1, 2)
    assert m.fixVar(x, Fraction(3, 2)) == (False, True)
    assert x.getLbLocal() == x.getUbLocal() == Fraction(3, 2)
    for name in RAISING:
        with pytest.raises(ValueError):
            getattr(m, name)(x, 1)
    with pytest.raises(ValueError):
        m.chgVarLbNode(None, x, 1)
    with pytest.raises(ValueError):
        m.chgVarUbNode(None, x, 1)
    assert not hasattr(m, "chgVarLbExact")


def test_addConsExactLinear():
    m = ExactModel()
    x = m.addVar("x")
    y = m.addVar("y")

    c = m.addConsExactLinear([x, y], [Fraction(1, 3), "1/10"], None, Fraction(2, 3), name="c")
    assert c.name == "c"
    assert m.getLhsExactLinear(c) == -float("inf")
    assert m.getRhsExactLinear(c) == Fraction(2, 3)
    assert m.getValsExactLinear(c) == {"x": Fraction(1, 3), "y": Fraction(1, 10)}

    m.chgCoefExactLinear(c, y, Fraction(1, 7))
    m.chgLhsExactLinear(c, 0)
    m.chgRhsExactLinear(c, None)
    z = m.addVar("z")
    m.addCoefExactLinear(c, z, 3)
    assert m.getValsExactLinear(c) == {"x": Fraction(1, 3), "y": Fraction(1, 7), "z": 3}
    assert m.getLhsExactLinear(c) == 0
    assert m.getRhsExactLinear(c) == float("inf")

    with pytest.raises(ValueError):
        m.addConsExactLinear([x, y], [1], 0, 1)
    with pytest.raises(ValueError):
        m.chgCoefExactLinear(c, x, float("inf"))
    with pytest.raises(Exception):
        m.addCons(x + y <= 1)

    n = Model()
    w = n.addVar("w")
    lin = n.addCons(w <= 1)
    with pytest.raises(ValueError):
        n.getValsExactLinear(lin)


def test_exact_solution():
    m = ExactModel()
    m.hideOutput()
    x = m.addVar("x", ub="10/3", obj=-1)
    y = m.addVar("y", vtype="I", ub=5, obj=Fraction(-1, 7))
    c = m.addConsExactLinear([x, y], [Fraction(1, 3), Fraction(1, 10)], None, Fraction(2, 3))
    m.addOrigObjoffsetExact(Fraction(1, 2))
    assert m.getOrigObjoffsetExact() == Fraction(1, 2)

    with pytest.raises(Warning):
        m.getPrimalboundExact()
    with pytest.raises(ValueError):
        m.addOrigObjoffsetExact(float("inf"))

    m.optimize()
    assert m.getStatus() == "optimal"

    sol = m.getBestSol()
    assert sol.isExact()
    assert m.getSolOrigObjExact(sol) == Fraction(-3, 2)
    assert m.getPrimalboundExact() == Fraction(-3, 2)
    assert m.getDualboundExact() == Fraction(-3, 2)
    assert m.getActivityExactLinear(c, sol) == Fraction(2, 3)
    assert m.getObjVal() == pytest.approx(-1.5)
