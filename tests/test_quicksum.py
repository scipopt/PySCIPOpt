from pyscipopt import Model, quicksum

def test_quicksum():
    m = Model("quicksum")
    x = m.addVar("x")
    y = m.addVar("y")
    z = m.addVar("z")
    c = 2.3

    s = quicksum([x,y,z,c])

    assert(s == sum([x,y,z,c]))
