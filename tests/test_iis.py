import pytest

from pyscipopt import Model, SCIP_RESULT, IISfinder

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
    def __init__(self, skip=False):
        self.skip = skip
        self.called = False
    
    def iisfinderexec(self):
        self.called = True
        if self.skip:
            self.iis.setSubscipInfeasible(True)
            self.iis.setSubscipIrreducible(False)
            return {"result": SCIP_RESULT.SUCCESS} # success to attempt to skip further processing

        subscip = self.iis.getSubscip()
        for c in subscip.getConss():
            if c.name in ["c2", "c4"]:
                subscip.delCons(c)

        self.iis.setSubscipInfeasible(True)
        self.iis.setSubscipIrreducible(True)
        return {"result": SCIP_RESULT.SUCCESS}

def test_custom_iis_finder():

    m = infeasible_model()
    my_iis = myIIS()

    m.setParam("iis/irreducible", False)
    m.setParam("iis/greedy/priority", -1000000) # lowering priority of greedy iis finder
    m.includeIISfinder(my_iis, "", "")

    m.generateIIS()
    assert my_iis.called

    iis = m.getIIS()
    assert iis.isSubscipIrreducible()
    assert iis.isSubscipInfeasible()
    subscip = iis.getSubscip()
    assert subscip.getNConss() == 2

def test_iisGreddyMakeIrreducible():
    m = infeasible_model()
    m.setParam("iis/irreducible", False)
    m.setParam("iis/greedy/priority", 1) # lowering priority of greedy iis finder

    my_iis = myIIS(skip=True)
    m.includeIISfinder(my_iis, "", "", priority=10000)

    iis = m.generateIIS()
    assert not iis.isSubscipIrreducible()
    assert iis.isSubscipInfeasible()

    iis.greedyMakeIrreducible()
    assert iis.isSubscipIrreducible()
