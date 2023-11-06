from pyscipopt import Model

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
    assert w.vtype() == "IMPLINT"

    m.chgVarType(x, 'I')
    assert x.vtype() == "INTEGER"

    m.chgVarType(y, 'M')
    assert y.vtype() == "IMPLINT"