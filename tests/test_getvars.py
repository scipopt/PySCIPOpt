import pytest

from pyscipopt import Model
from pyscipopt import quicksum

def setModel():
    m = Model()
    m.hideOutput()
    v0 = m.addVar("v0","B")
    v1 = m.addVar("v1","I")
    v2 = m.addVar("v2","C", ub=3.4)
    v3 = m.addVar("v3","BINARY")
    v4 = m.addVar("v4","INTEGER", ub=2)
    v5 = m.addVar("v5","CONTINUOUS", ub=1.3)
    v6 = m.addVar("v6","B")
    v7 = m.addVar("v7","B")
    v8 = m.addVar("v8","B")
    v9 = m.addVar(ub=0.6)
    vs = m.getVars()
    m.addCons(quicksum(vs) >= 11.5)
    m.setObjective(quicksum(vs))
    m.optimize()
    return m, vs

@pytest.mark.parametrize("idxvar", [0,1,[3,4,6],[],None])
def test_getvars(idxvar):
    m, vs = setModel()
    try:
        vs_ = [vs[i] for i in idxvar]
    except TypeError:
        if idxvar is None:
            vs_ = idxvar
        else:
            vs_ = vs[idxvar]
    print(m.getVals(vs_))
