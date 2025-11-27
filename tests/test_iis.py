import pytest

from pyscipopt import Model, SCIP_RESULT, IISfinder
from pyscipopt.recipes.infeasibilities import get_infeasible_constraints

def infeasible_model():
    m = Model()
    x1 = m.addVar("x1", vtype="B")
    x2 = m.addVar("x2", vtype="B")
    x3 = m.addVar("x3", vtype="B")

    m.addCons(x1 + x2 == 1, name="c1")
    m.addCons(x2 + x3 == 1, name="c2")
    m.addCons(x1 + x3 == 1, name="c3")
    m.addCons(x1 + x2 + x3 <= 0, name="c4")

    return m

def test_generate_iis():
    m = infeasible_model()

    m.optimize()

    # make sure IIS generation doesn't raise any exceptions
    iis = m.generateIIS()
    subscip = iis.getSubscip()
    assert iis.isSubscipIrreducible()
    assert subscip.getNConss() == 2
    assert iis.getNNodes() == 0
    assert m.isGE(iis.getTime(), 0)

class myIIS(IISfinder):
    def __init__(self, model, skip=False):
        super().__init__()
        self.model = model
        self.size = 0
        self.iis = None
        self.skip = skip
        self.called = False
    
    def iisfinderexec(self):
        self.called = True
        if self.skip:
            return {"result": SCIP_RESULT.SUCCESS} # success to attempt to skip further processing

        n_infeasibilities, _ = get_infeasible_constraints(self.model.__repr__.__self__)
        if n_infeasibilities == 0:
            return {"result": SCIP_RESULT.DIDNOTFIND}

        self.size = n_infeasibilities
        return {"result": SCIP_RESULT.SUCCESS}

def test_custom_iis_finder():

    m = infeasible_model()
    my_iis = myIIS(m)

    m.setParam("iis/irreducible", False)
    m.includeIISfinder(my_iis, "", "")

    m.generateIIS()
    assert my_iis.called

    iis = m.getIIS()
    iis.setSubscipInfeasible(True)
    subscip = iis.getSubscip()
    assert subscip.getNConss() == my_iis.size

def test_iisGreddyMakeIrreducible():
    m = infeasible_model()
    m.setParam("iis/irreducible", False)
    m.setParam("iis/greedy/timelimperiter", 0) # disabling greedy iis finder

    my_iis = myIIS(m, skip=True)
    m.includeIISfinder(my_iis, "", "", priority=99999999)
    m.optimize()

    iis = m.generateIIS()
    iis.setSubscipInfeasible(True)
    assert not iis.isSubscipIrreducible()

    iis.greedyMakeIrreducible()
    assert iis.isSubscipIrreducible()