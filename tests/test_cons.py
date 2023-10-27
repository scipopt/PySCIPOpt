from pyscipopt import Model, quicksum
import random


def test_getConsNVars():
    n_vars = random.randint(100,1000)
    m = Model()
    x = {}
    for i in range(n_vars):
        x[i] = m.addVar("%i"%i)
    
    c = m.addCons(quicksum(x[i] for i in x) <= 10)
    assert m.getConsNVars(c) == n_vars

    m.optimize()
    assert m.getConsNVars(c) == n_vars

def test_getConsVars():
    n_vars = random.randint(100,1000)
    m = Model()
    x = {}
    for i in range(n_vars):
        x[i] = m.addVar("%i"%i)
    
    c = m.addCons(quicksum(x[i] for i in x) <= 1)
    assert m.getConsVars(c) == [x[i] for i in x]
    