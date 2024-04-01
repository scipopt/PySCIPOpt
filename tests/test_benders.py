"""
flp-benders.py:  model for solving the capacitated facility location problem using Benders' decomposition

minimize the total (weighted) travel cost from n customers
to some facilities with fixed costs and capacities.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict, SCIP_PARAMSETTING
import pdb

def flp(I,J,d,M,f,c):
    """flp -- model for the capacitated facility location problem
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
    Returns a model, ready to be solved.
    """

    master = Model("flp-master")
    subprob = Model("flp-subprob")

    # creating the problem
    y = {}
    for j in J:
        y[j] = master.addVar(vtype="B", name="y(%s)"%j)

    master.setObjective(
        quicksum(f[j]*y[j] for j in J),
        "minimize")
    master.data = y

    # creating the subproblem
    x,y = {},{}
    for j in J:
        y[j] = subprob.addVar(vtype="B", name="y(%s)"%j)
        for i in I:
            x[i,j] = subprob.addVar(vtype="C", name="x(%s,%s)"%(i,j))

    for i in I:
        subprob.addCons(quicksum(x[i,j] for j in J) == d[i], "Demand(%s)"%i)

    for j in M:
        subprob.addCons(quicksum(x[i,j] for i in I) <= M[j]*y[j], "Capacity(%s)"%i)

    for (i,j) in x:
        subprob.addCons(x[i,j] <= d[i]*y[j], "Strong(%s,%s)"%(i,j))

    subprob.setObjective(
        quicksum(c[i,j]*x[i,j] for i in I for j in J),
        "minimize")
    subprob.data = x,y

    return master, subprob


def make_data():
    I,d = multidict({1:80, 2:270, 3:250, 4:160, 5:180})            # demand
    J,M,f = multidict({1:[500,1000], 2:[500,1000], 3:[500,1000]}) # capacity, fixed costs
    c = {(1,1):4,  (1,2):6,  (1,3):9,    # transportation costs
         (2,1):5,  (2,2):4,  (2,3):7,
         (3,1):6,  (3,2):3,  (3,3):4,
         (4,1):8,  (4,2):5,  (4,3):3,
         (5,1):10, (5,2):8,  (5,3):4,
         }
    return I,J,d,M,f,c


def test_flpbenders():
    '''
    test the Benders' decomposition plugins with the facility location problem.
    '''
    I,J,d,M,f,c = make_data()
    master, subprob = flp(I,J,d,M,f,c)
    # initializing the default Benders' decomposition with the subproblem
    master.setPresolve(SCIP_PARAMSETTING.OFF)
    master.setBoolParam("misc/allowstrongdualreds", False)
    master.setBoolParam("benders/copybenders", False)
    master.initBendersDefault(subprob)

    # optimizing the problem using Benders' decomposition
    master.optimize()

    # solving the subproblems to get the best solution
    master.computeBestSolSubproblems()

    EPS = 1.e-6
    y = master.data
    facilities = [j for j in y if master.getVal(y[j]) > EPS]

    x, suby = subprob.data
    edges = [(i,j) for (i,j) in x if subprob.getVal(x[i,j]) > EPS]

    print("Optimal value:", master.getObjVal())
    print("Facilities at nodes:", facilities)
    print("Edges:", edges)

    master.printStatistics()

    # since computeBestSolSubproblems() was called above, we need to free the
    # subproblems. This must happen after the solution is extracted, otherwise
    # the solution will be lost
    master.freeBendersSubproblems()

    assert 5.61e+03 - 10e-6 < master.getObjVal() < 5.61e+03 + 10e-6
