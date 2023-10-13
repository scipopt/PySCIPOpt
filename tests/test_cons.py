from pyscipopt import Model


def test_getConsVars():
    m = Model()
    x = m.addVar("x")
    y = m.addVar("y")
    c = m.addCons(x+y <= 1)
    assert m.getConsNVars(c) == len([x,y])
    assert len(m.getConsVars(c)) == len([x,y])
    