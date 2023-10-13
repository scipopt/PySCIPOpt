from pyscipopt import Model


def test_getConsNVars():
    m = Model()
    x = m.addVar()
    c = m.addCons(x == 1)
    assert m.getConsNVars(c)

