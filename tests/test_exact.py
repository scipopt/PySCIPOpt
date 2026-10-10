from fractions import Fraction

import pytest

from pyscipopt import Model


def _exact_supported():
    try:
        Model(exact=True)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _exact_supported(), reason="SCIP was built without exact solving support")


def test_model_exact():
    m = Model(exact=True)
    assert m.isExact()
    assert not Model().isExact()

    with pytest.raises(ValueError):
        Model(exact=True, createscip=False)
    with pytest.raises(ValueError):
        Model(sourceModel=m, exact=True)


def test_addVarExact():
    m = Model(exact=True)

    x = m.addVarExact("x", lb="1/3", ub=Fraction(10, 3), obj=-1)
    assert x.isExact()
    assert x.getLbOriginalExact() == Fraction(1, 3)
    assert x.getUbOriginalExact() == Fraction(10, 3)
    assert x.getLbGlobalExact() == Fraction(1, 3)
    assert x.getUbLocalExact() == Fraction(10, 3)
    assert x.getObjExact() == -1
    assert x.getLbOriginal() <= 1 / 3 <= x.getLbOriginal() + 1e-15

    y = m.addVarExact("y", lb=None, ub=None)
    assert y.getLbOriginalExact() == -float("inf")
    assert y.getUbOriginalExact() == float("inf")

    b = m.addVarExact("b", vtype="B")
    assert b.getUbOriginalExact() == 1
    assert b.vtype() == "BINARY"

    m.chgVarObjExact(x, Fraction(-1, 7))
    m.chgVarLbExact(x, None)
    m.chgVarUbExact(x, "5/2")
    assert x.getObjExact() == Fraction(-1, 7)
    assert x.getLbOriginalExact() == -float("inf")
    assert x.getUbOriginalExact() == Fraction(5, 2)

    with pytest.raises(ValueError):
        m.addVarExact("bad", lb=2, ub=1)
    with pytest.raises(Warning):
        m.addVarExact("bad", vtype="Q")
    with pytest.raises(ValueError):
        m.addVarExact("bad", lb="abc")

    with pytest.raises(Exception):
        Model().addVarExact("x")
    with pytest.raises(Exception):
        m.addVar("z")

    n = Model()
    z = n.addVar("z")
    assert not z.isExact()
    with pytest.raises(ValueError):
        z.getObjExact()


def test_addConsExactLinear():
    m = Model(exact=True)
    x = m.addVarExact("x")
    y = m.addVarExact("y")

    c = m.addConsExactLinear([x, y], [Fraction(1, 3), "1/10"], None, Fraction(2, 3), name="c")
    assert c.name == "c"
    assert m.getLhsExactLinear(c) == -float("inf")
    assert m.getRhsExactLinear(c) == Fraction(2, 3)
    assert m.getValsExactLinear(c) == {"x": Fraction(1, 3), "y": Fraction(1, 10)}

    m.chgCoefExactLinear(c, y, Fraction(1, 7))
    m.chgLhsExactLinear(c, 0)
    m.chgRhsExactLinear(c, None)
    z = m.addVarExact("z")
    m.addCoefExactLinear(c, z, 3)
    assert m.getValsExactLinear(c) == {"x": Fraction(1, 3), "y": Fraction(1, 7), "z": 3}
    assert m.getLhsExactLinear(c) == 0
    assert m.getRhsExactLinear(c) == float("inf")

    with pytest.raises(ValueError):
        m.addConsExactLinear([x, y], [1], 0, 1)
    with pytest.raises(Exception):
        m.addCons(x + y <= 1)

    n = Model()
    w = n.addVar("w")
    lin = n.addCons(w <= 1)
    with pytest.raises(ValueError):
        n.getValsExactLinear(lin)


def test_exact_solution():
    m = Model(exact=True)
    m.hideOutput()
    x = m.addVarExact("x", ub="10/3", obj=-1)
    y = m.addVarExact("y", vtype="I", ub=5, obj=Fraction(-1, 7))
    c = m.addConsExactLinear([x, y], [Fraction(1, 3), Fraction(1, 10)], None, Fraction(2, 3))
    m.addOrigObjoffsetExact(Fraction(1, 3))
    assert m.getOrigObjoffsetExact() == Fraction(1, 3)

    with pytest.raises(Warning):
        m.getPrimalboundExact()

    m.optimize()
    assert m.getStatus() == "optimal"

    sol = m.getBestSol()
    assert sol.isExact()
    assert m.getSolOrigObjExact(sol) == Fraction(-5, 3)
    assert m.getSolOrigObjExact(sol) != -1.6666666666666667
    assert m.getPrimalboundExact() == Fraction(-5, 3)
    assert m.getDualboundExact() == Fraction(-5, 3)
    assert m.getActivityExactLinear(c, sol) == Fraction(2, 3)
    assert m.getObjVal() == pytest.approx(-5 / 3)

    with pytest.raises(Warning):
        m.addOrigObjoffsetExact(1)
