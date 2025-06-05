from pyscipopt import Model, Decomposition, quicksum
import pytest


def test_consLabels():
    m = Model()
    a = m.addVar("C", lb = 0)
    b = m.addVar("C", lb = 0)
    c = m.addVar("M", lb = 0)
    d = m.addVar("C", lb = 0)
     
    cons_1 = m.addCons( a + b <= 5)
    cons_2a = m.addCons( c**2 + d**2 <= 8)
    cons_2b = m.addCons( c == d )
    decomp = Decomposition()
    decomp.setConsLabels({cons_1: 1, cons_2a: 2, cons_2b: 2})

    m.addDecomposition(decomp)
    o = m.addVar("C")
    m.addCons( o <= a + b + c + d)
    m.setObjective("o", "minimize")
    m.optimize()
    
test_consLabels()
