from pyscipopt import Model, SCIP_RESULT
from pyscipopt.scip import Relax
import pytest 

calls = []


class SoncRelax(Relax):
    def relaxexec(self):
        calls.append('relaxexec')
        return {
            'result': SCIP_RESULT.SUCCESS,
            'lowerbound': 10e4
        }


def test_relax():
    m = Model()
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
    assert m.getObjVal() > 10e4

class EmptyRelaxator(Relax):
    def relaxexec(self):
        pass
        # doesn't return anything

def test_empty_relaxator():
    m = Model()
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
        m.optimize()
