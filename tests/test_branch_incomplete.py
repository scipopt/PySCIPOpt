import pytest
import os
from pyscipopt import Model, Branchrule, SCIP_PARAMSETTING


def test_incomplete_branchrule():
    class IncompleteBranchrule(Branchrule):
        pass

    branchrule = IncompleteBranchrule()
    model = Model()
    model.addVar(obj=1, lb=0, vtype="INTEGER")
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.includeBranchrule(branchrule, "", "", priority=10000000, maxdepth=-1, maxbounddist=1)

    with pytest.raises(Exception):
        model.optimize()
