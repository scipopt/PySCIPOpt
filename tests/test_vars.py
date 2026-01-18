from pyscipopt import Model, SCIP_PARAMSETTING, SCIP_BRANCHDIR, SCIP_IMPLINTTYPE
from helpers.utils import random_mip_1

def test_variablebounds():
    m = Model()

    x0 = m.addVar(lb=-5, ub=8)
    r1 = m.addVar()
    r2 = m.addVar()
    y0 = m.addVar(lb=3)
    t = m.addVar(lb=None)
    z = m.addVar()

    m.chgVarLbGlobal(x0, -2)
    m.chgVarUbGlobal(x0, 4)

    infeas, tightened = m.tightenVarLb(x0, -5)
    assert not infeas
    assert not tightened
    infeas, tightened = m.tightenVarLbGlobal(x0, -1)
    assert not infeas
    assert tightened
    infeas, tightened = m.tightenVarUb(x0, 3)
    assert not infeas
    assert tightened
    infeas, tightened = m.tightenVarUbGlobal(x0, 9)
    assert not infeas
    assert not tightened
    infeas, fixed = m.fixVar(z, 7)
    assert not infeas
    assert fixed
    assert m.delVar(z)

    m.addCons(r1 >= x0)
    m.addCons(r2 >= -x0)
    m.addCons(y0 == r1 +r2)

    m.setObjective(t)
    m.addCons(t >= r1 * (r1 - x0) + r2 * (r2 + x0))


    m.optimize()

    print("x0", m.getVal(x0))
    print("r1", m.getVal(r1))
    print("r2", m.getVal(r2))
    print("y0", m.getVal(y0))
    print("t", m.getVal(t))

def test_vtype():
    m = Model()

    x = m.addVar(vtype= 'C', lb=-5.5, ub=8)
    y = m.addVar(vtype= 'I', lb=-5.2, ub=8)
    z = m.addVar(vtype= 'B', lb=-5.2, ub=8)
    w = m.addVar(vtype= 'M', lb=-5.2, ub=8)

    assert x.vtype() == "CONTINUOUS"
    assert y.vtype() == "INTEGER"
    assert z.vtype() == "BINARY"
    assert w.vtype() == "CONTINUOUS" 

    is_int = lambda x: x.isIntegral()
    is_implint = lambda x: x.isImpliedIntegral()
    # is_nonimplint = lambda x: x.isNonImpliedIntegral()
    is_bin = lambda x: x.isBinary()

    assert not is_int(x) and not is_implint(x) and not is_bin(x)
    assert is_int(y) and not is_implint(y) and not is_bin(y)
    assert is_int(z) and not is_implint(z)  and is_bin(z)
    assert w.vtype() == "CONTINUOUS" and is_int(w) and is_implint(w) and not is_bin(w)

    assert w.getImplType() == SCIP_IMPLINTTYPE.WEAK

    m.chgVarType(x, 'I')
    assert x.vtype() == "INTEGER"

def test_markRelaxationOnly():
    m = Model()

    x = m.addVar(vtype='C', lb=-5.5, ub=8, deletable=True)
    y = m.addVar(vtype='I', lb=-5.2, ub=8)

    assert not x.isRelaxationOnly()
    assert not y.isRelaxationOnly()

    x.markRelaxationOnly()
    assert x.isRelaxationOnly()
    assert x.isDeletable()
    assert not y.isRelaxationOnly()
    assert not y.isDeletable()

def test_getNBranchings():
    m = random_mip_1(True, True, True, 100, True)
    m.setParam("branching/mostinf/priority", 999999)
    m.setParam("limits/restarts", 0)

    m.optimize()

    m.setParam("limits/nodes", 200)
    m.restartSolve()
    m.optimize()

    n_branchings = 0
    for var in m.getVars():
        n_branchings += var.getNBranchings(SCIP_BRANCHDIR.UPWARDS)
        n_branchings += var.getNBranchings(SCIP_BRANCHDIR.DOWNWARDS)

    assert n_branchings == m.getNTotalNodes() - 2 # "-2" comes from the two root nodes because of the restart

def test_getNBranchingsCurrentRun():
    m = random_mip_1(True, True, True, 100, True)
    m.setParam( "branching/mostinf/priority", 999999)

    m.optimize()

    n_branchings = 0
    for var in m.getVars():
        n_branchings += var.getNBranchingsCurrentRun(SCIP_BRANCHDIR.UPWARDS)
        n_branchings += var.getNBranchingsCurrentRun(SCIP_BRANCHDIR.DOWNWARDS)

    assert n_branchings == m.getNNodes() - 1

def test_isActive():
    m = Model()
    x = m.addVar(vtype='C', lb=0.0, ub=1.0)
    # newly added variables should be active
    assert x.isActive()
    m.freeProb()

    # TODO lacks tests for cases when returned false due to 
    # - fixed (probably during probing)
    # - aggregated

def test_markDoNotAggrVar_and_getStatus():
    model = Model()
    x = model.addVar("x", obj=2, lb=0, ub=10)
    y = model.addVar("y", obj=3, lb=0, ub=20)
    z = model.addVar("z", obj=1, lb=0, ub=10)
    w = model.addVar("w", obj=4, lb=0, ub=15)

    model.addCons(y - 2*x == 0)
    model.addCons(x + z + w == 10)
    model.addCons(x*y*z >= 21) # to prevent presolve from removing all variables
    model.presolve()

    assert z.getStatus() == "ORIGINAL"
    assert model.getTransformedVar(z).getStatus() == "AGGREGATED"
    assert model.getTransformedVar(w).getStatus() == "MULTAGGR"

    assert model.getNVars(True) == 1

    model.freeTransform()
    model.markDoNotMultaggrVar(w)
    model.presolve()

    assert model.getTransformedVar(w).getStatus() != "MULTAGGR"
    assert model.getNVars(True) == 3

    model.freeTransform()
    model.markDoNotAggrVar(y)
    model.presolve()
    assert model.getTransformedVar(z).getStatus() != "AGGREGATED"
    assert model.getNVars(True) == 4

    assert x.getStatus() == "ORIGINAL"


def test_isIntegral():
    """Test that Model.isIntegral correctly identifies integral values."""
    m = Model()

    # Exact integer values should be integral
    assert m.isIntegral(5.0)
    assert m.isIntegral(0.0)
    assert m.isIntegral(-3.0)

    # Values very close to integers (within epsilon) should be integral
    eps = m.epsilon()
    assert m.isIntegral(5.0 + eps / 2)
    assert m.isIntegral(5.0 - eps / 2)

    # Non-integer values should not be integral
    assert not m.isIntegral(5.5)
    assert not m.isIntegral(0.1)
    assert not m.isIntegral(-3.7)


def test_adjustedVarLb():
    """Test that Model.adjustedVarLb correctly rounds bounds for integer variables."""
    m = Model()

    # For integer variables, lower bounds should be rounded up (ceiling)
    x_int = m.addVar(vtype='I', lb=-10.0, ub=10.0, name="x_int")
    # 2.3 should be adjusted to 3.0 for lower bound of integer
    assert m.adjustedVarLb(x_int, 2.3) == 3.0
    # -2.3 should be adjusted to -2.0 for lower bound of integer
    assert m.adjustedVarLb(x_int, -2.3) == -2.0
    # Exact integer should stay the same
    assert m.adjustedVarLb(x_int, 5.0) == 5.0

    # For continuous variables, values within epsilon of zero should be set to 0
    x_cont = m.addVar(vtype='C', lb=-10.0, ub=10.0, name="x_cont")
    eps = m.epsilon()
    assert m.adjustedVarLb(x_cont, eps / 2) == 0.0
    # Non-zero values should stay the same
    assert m.adjustedVarLb(x_cont, 5.5) == 5.5


def test_adjustedVarUb():
    """Test that Model.adjustedVarUb correctly rounds bounds for integer variables."""
    m = Model()

    # For integer variables, upper bounds should be rounded down (floor)
    x_int = m.addVar(vtype='I', lb=-10.0, ub=10.0, name="x_int")
    # 2.7 should be adjusted to 2.0 for upper bound of integer
    assert m.adjustedVarUb(x_int, 2.7) == 2.0
    # -2.7 should be adjusted to -3.0 for upper bound of integer
    assert m.adjustedVarUb(x_int, -2.7) == -3.0
    # Exact integer should stay the same
    assert m.adjustedVarUb(x_int, 5.0) == 5.0

    # For continuous variables, values should generally stay the same
    x_cont = m.addVar(vtype='C', lb=-10.0, ub=10.0, name="x_cont")
    assert m.adjustedVarUb(x_cont, 5.5) == 5.5
    assert m.adjustedVarUb(x_cont, -3.2) == -3.2