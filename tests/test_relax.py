from pyscipopt import Model, SCIP_RESULT, SCIP_PARAMSETTING
from pyscipopt.scip import Relax
import pytest 
from helpers.utils import random_mip_1

calls = []


class SoncRelax(Relax):
    def relaxexec(self):
        calls.append('relaxexec')
        return {
            'result': SCIP_RESULT.SUCCESS,
            'lowerbound': 10e4
        }


def test_relaxator():
    m = Model()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.setSeparating(SCIP_PARAMSETTING.OFF)
    m.setParam("limits/nodes", 1)
    m.hideOutput()

    # include relaxator
    m.includeRelax(SoncRelax(), 'testrelaxator',
                   'Test that relaxator gets included')

    # add Variables
    x0 = m.addVar(vtype="I", name="x0")
    x1 = m.addVar(vtype="I", name="x1")
    x2 = m.addVar(vtype="I", name="x2")

    # addCons
    m.addCons(x0 >= 2)
    m.addCons(x0**2 <= x1)
    m.addCons(x1 * x2 >= x0)

    m.setObjective(x1 + x0)
    m.optimize()

    assert 'relaxexec' in calls
    assert len(calls) >= 1
    assert m.isGE(m.getDualbound(), 10e4)

class EmptyRelaxator(Relax):
    pass

def test_empty_relaxator():
    m = Model()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.hideOutput()

    m.includeRelax(EmptyRelaxator(), "", "")

    x0 = m.addVar(vtype="I", name="x0")
    x1 = m.addVar(vtype="I", name="x1")
    x2 = m.addVar(vtype="I", name="x2")

    m.addCons(x0 >= 2)
    m.addCons(x0**2 <= x1)
    m.addCons(x1 * x2 >= x0)

    m.setObjective(x1 + x0)

    with pytest.raises(Exception):
        with pytest.raises(AssertionError):
            m.optimize()

def test_relax():
    model = random_mip_1()

    x = model.addVar(vtype="B")

    model.relax()

    assert x.getLbGlobal() == 0 and x.getUbGlobal() == 1

    for var in model.getVars():
        assert var.vtype() == "CONTINUOUS"