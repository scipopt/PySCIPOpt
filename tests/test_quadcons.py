from pyscipopt import Model

def test_niceqp():
    s = Model()

    x = s.addVar("x")
    y = s.addVar("y")
    s.addCons(x >= 2)
    s.addCons(x*x <= y)
    s.setObjective(y, sense='minimize')

    s.optimize()

    assert round(s.getVal(x)) == 2.0
    assert round(s.getVal(y)) == 4.0

def test_niceqcqp():
    s = Model()

    x = s.addVar("x")
    y = s.addVar("y")
    s.addCons(x*x + y*y <= 2)
    s.setObjective(x + y, sense='maximize')

    s.optimize()

    assert round(s.getVal(x)) == 1.0
    assert round(s.getVal(y)) == 1.0