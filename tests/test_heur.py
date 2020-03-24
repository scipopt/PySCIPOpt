import gc
import weakref

import pytest

from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_PARAMSETTING, SCIP_HEURTIMING
from pyscipopt.scip import is_memory_freed

from util import is_optimized_mode

class MyHeur(Heur):

    def heurexec(self, heurtiming, nodeinfeasible):

        sol = self.model.createSol(self)
        vars = self.model.getVars()

        sol[vars[0]] = 5.0
        sol[vars[1]] = 0.0

        accepted = self.model.trySol(sol)

        if accepted:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}

def test_heur():
    # create solver instance
    s = Model()
    heuristic = MyHeur()
    s.includeHeur(heuristic, "PyHeur", "custom heuristic implemented in python", "Y", timingmask=SCIP_HEURTIMING.BEFORENODE)
    s.setPresolve(SCIP_PARAMSETTING.OFF)

    # add some variables
    x = s.addVar("x", obj=1.0)
    y = s.addVar("y", obj=2.0)

    # add some constraint
    s.addCons(x + 2*y >= 5)

    # solve problem
    s.optimize()

    # print solution
    sol = s.getBestSol()
    assert sol != None
    assert round(sol[x]) == 5.0
    assert round(sol[y]) == 0.0

def test_heur_memory():
    if is_optimized_mode():
       pytest.skip()

    def inner():
        s = Model()
        heuristic = MyHeur()
        s.includeHeur(heuristic, "PyHeur", "custom heuristic implemented in python", "Y", timingmask=SCIP_HEURTIMING.BEFORENODE)
        return weakref.proxy(heuristic)

    heur_prox = inner()
    gc.collect() # necessary?
    with pytest.raises(ReferenceError):
        heur_prox.name

    assert is_memory_freed()

if __name__ == "__main__":
    test_heur()
