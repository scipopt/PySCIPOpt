import pytest

from pyscipopt import Model
from pyscipopt.scip import IISfinder

calls = []
class myIISfinder(IISfinder):
    def iisfinderexec(self):
        calls.append('relaxexec')

def test_iis_custom():
    from helpers.utils import random_mip_1

    m = random_mip_1()
    x = m.addVar()
    m.addCons(x >= 1, "inf1")
    m.addCons(x <= 0, "inf2")

    iis = myIISfinder()
    m.includeIISfinder(iis, name="custom", desc="test")
    m.optimize()
    assert calls != []

def test_iis_greedy():
    m = Model()
    x = m.addVar()
    m.addCons(x >= 1, "inf1")
    m.addCons(x <= 0, "inf2")

    m.includeIISfinderGreedy()
    m.optimize()

test_iis_greedy()
test_iis_custom()
