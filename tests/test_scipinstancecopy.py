from pyscipopt import Model

def test_scipinstancecopy():
    # create solver instance
    s = Model()

    # add some variables
    x = s.addVar("x", vtype = 'C', obj = 1.0)
    y = s.addVar("y", vtype = 'C', obj = 2.0)
    s.setObjective(4.0 * y, clear = False)

    c = s.addCons(x + 2 * y >= 1.0)

    # solve problems
    s.optimize()

    s2 = s.createModelFromSCIP()

    assert s.getObjVal() == s2.getObjVal()

if __name__ == "__main__":
    test_scipinstancecopy()
