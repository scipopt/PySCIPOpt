import pytest
from pyscipopt import Model, Branchrule, SCIP_PARAMSETTING


@pytest.mark.skip(reason="fix later")
def test_incomplete_branchrule():
    class IncompleteBranchrule(Branchrule):
        pass

    branchrule = IncompleteBranchrule()
    model = Model()
    x = model.addVar(obj=-5, ub=100, vtype="INTEGER")
    y = model.addVar(obj=-6, ub=100, vtype="INTEGER")
    model.addCons(x + y <= 5)
    model.addCons(4*x + 7*y <= 28)
    model.includeBranchrule(branchrule, "", "", priority=99999999, maxdepth=-1, maxbounddist=1)
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.disablePropagation()

    with pytest.raises(Exception):
        model.optimize()
