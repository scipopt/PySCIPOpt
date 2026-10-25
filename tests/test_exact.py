from fractions import Fraction

import pytest

from pyscipopt import SCIP_PARAMSETTING, Model, ExactModel, ExactVariable, Variable


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
    with pytest.raises(ValueError):
        m.setObjlimit(Fraction(1, 3))
    assert not hasattr(m, "chgVarLbExact")


def test_linear_constraints():
    m = ExactModel()
    x = m.addVar("x")
    y = m.addVar("y")
    c = m.addConsLinear([x, y], [Fraction(1, 3), "1/10"], None, Fraction(2, 3), name="c")
    assert c.name == "c"
    assert m.getLhs(c) == -float("inf")
    assert m.getRhs(c) == Fraction(2, 3)
    assert m.getValsLinear(c) == {"x": Fraction(1, 3), "y": Fraction(1, 10)}
    assert m.getConsVals(c) == [Fraction(1, 3), Fraction(1, 10)]
    assert [v.name for v in m.getConsVars(c)] == ["x", "y"]
    assert all(isinstance(v, ExactVariable) for v in m.getConsVars(c))

    m.chgCoefLinear(c, y, Fraction(1, 7))
    m.chgLhs(c, 0)
    m.chgRhs(c, None)
    z = m.addVar("z")
    m.addCoefLinear(c, z, 3)
    assert m.getValsLinear(c) == {"x": Fraction(1, 3), "y": Fraction(1, 7), "z": 3}
    m.delCoefLinear(c, z)
    assert "z" not in m.getValsLinear(c)
    assert m.getLhs(c) == 0
    assert m.getRhs(c) == float("inf")

    with pytest.raises(ValueError):
        m.addConsLinear([x, y], [1], 0, 1)
    with pytest.raises(ValueError):
        m.chgCoefLinear(c, x, float("inf"))
    with pytest.raises(TypeError):
        m.addCons(x + y <= 1)
    with pytest.raises(TypeError):
        m.addCons(Fraction(1, 3) * x <= 1)

    m.addObjoffset(Fraction(1, 3))
    assert m.getObjoffset() == Fraction(1, 3)
    with pytest.raises(ValueError):
        m.addObjoffset(1, solutions=True)
    with pytest.raises(ValueError):
        m.addObjoffset(float("inf"))
    with pytest.raises(ValueError):
        m.getObjoffset(original=False)

    n = Model()
    w = n.addVar("w")
    lin = n.addCons(w <= 1)
    assert isinstance(n.getLhs(lin), float)
    assert not hasattr(n, "addConsLinear")
    assert not hasattr(m, "addConsExactLinear")


def _knapsack():
    m = ExactModel()
    m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    x = m.addVar("x", ub="10/3", obj=-1)
    y = m.addVar("y", vtype="I", ub=5, obj=Fraction(-1, 7))
    c = m.addConsLinear([x, y], [Fraction(1, 3), Fraction(1, 10)], None, Fraction(2, 3))
    m.addObjoffset(Fraction(1, 2))
    return m, x, y, c


def test_solutions():
    m, x, y, c = _knapsack()
    with pytest.raises(Warning):
        m.getPrimalbound()
    with pytest.raises(ValueError):
        m.createOrigSol()
    with pytest.raises(ValueError):
        m.createPartialSol()

    sol = m.createSol()
    m.setSolVal(sol, x, Fraction(1, 2))
    m.setSolVal(sol, y, 1)
    assert sol.isExact()
    assert m.getSolVal(sol, x) == Fraction(1, 2)
    with pytest.raises(TypeError):
        m.getSolVal(sol, x + y)
    assert m.addSol(sol) is True

    m.presolve()
    sol2 = m.createSol()
    m.setSolVal(sol2, x, 2)
    m.setSolVal(sol2, y, 0)
    assert m.trySol(sol2) is True

    m.optimize()
    assert m.getStatus() == "optimal"
    best = m.getBestSol()
    assert best.isExact()
    assert m.getObjVal() == Fraction(-3, 2)
    assert m.getSolObjVal(best) == Fraction(-3, 2)
    assert m.getPrimalbound() == m.getDualbound() == Fraction(-3, 2)
    assert isinstance(m.getUpperbound(), Fraction)
    assert isinstance(m.getLowerbound(), Fraction)
    assert isinstance(m.getCutoffbound(), (Fraction, float))
    assert m.getVal(x) == 2
    assert m.getVal(y) == 0
    assert m.getActivity(c, best) == Fraction(2, 3)
    assert m.getSlack(c, best) == 0
    m.printSol(best)
    m.printBestSol()
    best.retransform()
    with pytest.raises(Exception):
        m.addObjoffset(1)
    assert not hasattr(m, "getSolValExact")


def test_no_exact_names():
    for cls in (Model, Variable, ExactModel, ExactVariable):
        leaked = [n for n in dir(cls) if "Exact" in n and not n.startswith("_")]
        assert leaked == [], leaked
    m = ExactModel()
    assert m.allowNegSlack() in (True, False)
    assert not hasattr(Model(), "allowNegSlack")


def test_numerics():
    m = ExactModel()
    third = Fraction(1, 3)
    tiny = Fraction(1, 10**30)
    inf = m.infinity()
    assert inf == float("inf")
    assert m.epsilon() == 0 and m.feastol() == 0
    assert m.isEQ(third, "1/3") and m.isFeasEQ(third, Fraction(2, 6))
    assert not m.isEQ(third, third + tiny) and not m.isFeasEQ(third, third + tiny)
    assert m.isLT(third, third + tiny) and m.isFeasLT(third, third + tiny)
    assert m.isLE(third, third) and m.isFeasLE(third, third)
    assert not m.isGT(third, third) and m.isFeasGT(third + tiny, third)
    assert m.isGE(third, third) and not m.isFeasGE(third, third + tiny)
    assert m.isLT(third, inf) and m.isGT(third, -inf) and m.isEQ(inf, inf)
    assert m.isZero(0) and not m.isZero(tiny) and not m.isFeasZero(tiny)
    assert m.isPositive(tiny) and m.isFeasPositive(tiny)
    assert m.isNegative(-tiny) and m.isFeasNegative(-tiny)
    assert m.isIntegral(Fraction(4, 2)) and not m.isIntegral(1 + tiny) and not m.isFeasIntegral(1 + tiny)
    assert m.isInfinity(inf) and not m.isInfinity(-inf) and not m.isInfinity(10**30)
    assert m.frac(Fraction(7, 3)) == third and m.feasFrac(Fraction(-7, 3)) == Fraction(2, 3)
    assert m.feasFloor(Fraction(7, 3)) == 2 and m.feasCeil(Fraction(7, 3)) == 3
    assert m.feasRound(Fraction(1, 2)) == 1 and m.feasRound(Fraction(-1, 2)) == 0
    with pytest.raises(ValueError):
        m.isHugeValue(1)
