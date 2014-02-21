import pyscipopt.scip as scip

def test_simplelp():
    # create solver instance
    s = scip.Solver()
    s.create()
    s.includeDefaultPlugins()
    s.createProbBasic("Knapsack")

    # add some variables
    x = s.addContVar("x", obj=1.0)
    y = s.addContVar("y", obj=2.0)

    # add some constraint
    coeffs = {x: 1.0, y: 2.0}
    s.addCons(coeffs, 5.0)

    # solve problem
    s.solve()

    # retrieving the best solution
    solution = s.getBestSol()

    # print solution
    assert round(s.getVal(solution, x)) == 5.0
    assert round(s.getVal(solution, y)) == 0.0

    s.free()

if __name__ == '__main__':
    test_simplelp()
