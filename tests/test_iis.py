import pytest

from pyscipopt import Model, IISfinder

class myIIS(IISfinder):
    def __init__(self):
        super().__init__()
        self._iisfinder = None

    def isIISFound(self):
        return self._iisfinder is not None

def test_iis_greedy_make_irreducible():
    m = Model()
    x1 = m.addVar("x1")
    x2 = m.addVar("x2")
    x3 = m.addVar("x3")

    m.addCons(x1 + x2 >= 5)
    m.addCons(x2 + x3 >= 5)
    m.addCons(x1 + x3 <= 3)

    iisfinder = IISfinder()

    m.iisGreedyMakeIrreducible(iisfinder)

    assert iisfinder.isIISFound() == True

def test_custom_iis():
    m = Model()
    x1 = m.addVar("x1")
    x2 = m.addVar("x2")
    x3 = m.addVar("x3")

    m.addCons(x1 + x2 >= 5)
    m.addCons(x2 + x3 >= 5)
    m.addCons(x1 + x3 <= 3)

    iisfinder = myIIS()

    pass

