import pytest

from pyscipopt import Model, IISfinder

def infeasible_model():
    m = Model()
    x1 = m.addVar("x1", lb=0, ub=1, vtype="B")
    x2 = m.addVar("x2", lb=0, ub=1, vtype="B")
    x3 = m.addVar("x3", lb=0, ub=1, vtype="B")

    m.addCons(x1 + x2 == 1)
    m.addCons(x2 + x3 == 1)
    m.addCons(x1 + x3 == 1)

    return m

def test_generate_iis():
    m = infeasible_model()

    # make sure IIS generation doesn't raise any exceptions
    m.generateIIS()


def test_custom_iis_finder():
    class MyIIS(IISfinder):
        def __init__(self):
            super().__init__()
            self._iisfinder = None
        
    
    m = infeasible_model()
    my_iis = MyIIS()

    m.includeIISfinder(my_iis, "", "")
    
    # should raise an exception since the custom IIS finder doesn't implement the exec method
    with pytest.raises(Exception):
        m.generateIIS()
    

