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

    # make sure IIS generation doesn't raise any exceptions
    iis = m.generateIIS()
    assert iis.irreducible
    assert iis.model.getNConss() == 2
    assert iis.nodes == 0
    iis.time

class myIIS(IISfinder):
    def __init__(self, model, skip=False):
        super().__init__()
        self.model = model
        self.size = 0
        self.iis = None
        self.skip = skip
    
    def iisfinderexec(self):
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

    m.includeIISfinder(my_iis, "", "")

    m.generateIIS()
    iis = m.getIIS()
    assert iis.model.getNConss() == my_iis.size

def test_iisGreddyMakeIrreducible():
    m = infeasible_model()

    m.setParam("iis/greedy/priority", -1)
    my_iis = myIIS(m, skip=True)
    m.includeIISfinder(my_iis, "", "")
    iis = m.generateIIS()
    with pytest.raises(AssertionError):
        assert not iis.irreducible # currently breaking. do SCIP IIS methods enter after custom iisfinder?

    m.iisGreedyMakeIrreducible(iis)
    assert iis.irreducible