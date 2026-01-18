from pyscipopt import Model
from helpers.utils import random_mip_1

def test_copy():
    # create solver instance
    s = Model()

    # add some variables
    x = s.addVar("x", vtype = 'C', obj = 1.0)
    y = s.addVar("y", vtype = 'C', obj = 2.0)
    s.setObjective(4.0 * y, clear = False)

    c = s.addCons(x + 2 * y >= 1.0, modifiable = True)

    s2 = Model(sourceModel=s)

    # solve problems
    s.optimize()
    s2.optimize()

    assert s.getObjVal() == s2.getObjVal()
    assert s.getConss()[0].isModifiable() and s2.getConss()[0].isModifiable()

