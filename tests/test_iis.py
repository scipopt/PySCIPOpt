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

    return m

def test_generate_iis():
    m = infeasible_model()

    # make sure IIS generation doesn't raise any exceptions
    m.generateIIS()


class myIIS(IISfinder):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.size = 0
        self.iis = None
    
    def iisfinderexec(self):
        n_infeasibilities, aux_vars = get_infeasible_constraints(self.model.__repr__.__self__)
        if n_infeasibilities == 0:
            return {"result": SCIP_RESULT.DIDNOTFIND}

        self.size = n_infeasibilities
        self.iis = aux_vars
        return {"result": SCIP_RESULT.SUCCESS}

def test_custom_iis_finder():        
    
    m = infeasible_model()
    my_iis = myIIS(m)

    m.includeIISfinder(my_iis, "", "")

    m.generateIIS()
    assert my_iis.size == 1
