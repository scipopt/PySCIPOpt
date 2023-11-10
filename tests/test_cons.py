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
   
def test_constraint_option_setting():
    m = Model()
    x = m.addVar()
    c = m.addCons(x >= 3)

    for option in [True, False]:
        m.setCheck(c, option)
        m.setEnforced(c, option)
        m.setRemovable(c, option)
        m.setInitial(c, option)

        assert c.isChecked() == option
        assert c.isEnforced() == option
        assert c.isRemovable() == option
        assert c.isInitial() == option