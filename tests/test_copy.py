from pyscipopt import Model
from helpers.utils import random_mip_1

def test_copy():
    # create solver instance
    s = Model()

    # add some variables
    x = s.addVar("x", vtype = 'C', obj = 1.0)
    y = s.addVar("y", vtype = 'C', obj = 2.0)
    s.setObjective(4.0 * y, clear = False)

    c = s.addCons(x + 2 * y >= 1.0)

    s2 = Model(sourceModel=s)

    # solve problems
    s.optimize()
    s2.optimize()

    assert s.getObjVal() == s2.getObjVal()


def test_addOrigVarsConssObjectiveFrom():
    m = Model()
    x = m.addVar("x", vtype = 'B')
    y = m.addVar("y", vtype = 'B')
    m.addCons(x + y >= 1)
    m.addCons(x + y <= 2)
    m.setObjective(x + y, 'maximize')

    m1 = Model()
    m1.addOrigVarsConssObjectiveFrom(m)

    m.optimize()
    m1.optimize()

    assert m.getNVars(transformed=False) == m1.getNVars(transformed=False)
    assert m.getNConss(transformed=False) == m1.getNConss(transformed=False)
    assert m.getObjVal() == m1.getObjVal() == 2 

