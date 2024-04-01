import pytest
import os
from pyscipopt import Model, Branchrule, SCIP_PARAMSETTING

@pytest.mark.skip(reason="fix later")
def test_incomplete_branchrule():
    class IncompleteBranchrule(Branchrule):
        pass

    branchrule = IncompleteBranchrule()
    model = Model()
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.includeBranchrule(branchrule, "", "", priority=10000000, maxdepth=-1, maxbounddist=1)
    model.readProblem(os.path.join("tests", "data", "10teams.mps"))

    with pytest.raises(Exception):
        model.optimize()
