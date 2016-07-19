from pyscipopt import Model, quicksum

def test_quicksum():
    m = Model("quicksum")
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    c = 2.3

    q = quicksum([x,y,z,c]) == 0.0
    s =      sum([x,y,z,c]) == 0.0

    assert(q.expr.terms == s.expr.terms)
