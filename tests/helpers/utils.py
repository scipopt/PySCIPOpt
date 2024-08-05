from pyscipopt import Model, quicksum, SCIP_PARAMSETTING, exp, log, sqrt, sin
from typing import List

def random_MIP_1():
    scip = Model()
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    scip.setLongintParam("limits/nodes", 250)
    scip.setParam("presolving/maxrestarts", 0)

    x0 = scip.addVar(lb=-2, ub=4)
    r1 = scip.addVar()
    r2 = scip.addVar()
    y0 = scip.addVar(lb=3)
    t = scip.addVar(lb=None)
    l = scip.addVar(vtype="I", lb=-9, ub=18)
    u = scip.addVar(vtype="I", lb=-3, ub=99)

    more_vars = []
    for i in range(100):
        more_vars.append(scip.addVar(vtype="I", lb=-12, ub=40))
        scip.addCons(quicksum(v for v in more_vars) <= (40 - i) * quicksum(v for v in more_vars[::2]))

    for i in range(100):
        more_vars.append(scip.addVar(vtype="I", lb=-52, ub=10))
        scip.addCons(quicksum(v for v in more_vars[50::2]) <= (40 - i) * quicksum(v for v in more_vars[200::2]))

    scip.addCons(r1 >= x0)
    scip.addCons(r2 >= -x0)
    scip.addCons(y0 == r1 + r2)
    scip.addCons(t + l + 7 * u <= 300)
    scip.addCons(t >= quicksum(v for v in more_vars[::3]) - 10 * more_vars[5] + 5 * more_vars[9])
    scip.addCons(more_vars[3] >= l + 2)
    scip.addCons(7 <= quicksum(v for v in more_vars[::4]) - x0)
    scip.addCons(quicksum(v for v in more_vars[::2]) + l <= quicksum(v for v in more_vars[::4]))

    scip.setObjective(t - quicksum(j * v for j, v in enumerate(more_vars[20:-40])))

    return scip

def random_lp_1():
    return random_MIP_1().relax()
    

def random_nlp_1():
    model = Model()

    v = model.addVar()
    w = model.addVar()
    x = model.addVar()
    y = model.addVar()
    z = model.addVar()

    model.addCons(exp(v)+log(w)+sqrt(x)+sin(y)+z**3 * y <= 5)
    model.setObjective(v + w + x + y + z, sense='maximize')

    return model

def knapsack_model(weights = [4, 2, 6, 3, 7, 5], costs = [7, 2, 5, 4, 3, 4]):
    # create solver instance
    s = Model("Knapsack")
    s.hideOutput()

    # setting the objective sense to maximise
    s.setMaximize()

    assert len(weights) == len(costs)

    # knapsack size
    knapsackSize = 15

    # adding the knapsack variables
    knapsackVars = []
    varNames = []
    varBaseName = "Item"
    for i in range(len(weights)):
        varNames.append(varBaseName + "_" + str(i))
        knapsackVars.append(s.addVar(varNames[i], vtype='I', obj=costs[i], ub=1.0))


    # adding a linear constraint for the knapsack constraint
    s.addCons(quicksum(w*v for (w, v) in zip(weights, knapsackVars)) <= knapsackSize)

    return s

def bin_packing_model(sizes: List[int], capacity: int) -> Model:
    model = Model("Binpacking")
    n = len(sizes)
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = model.addVar(vtype="B", name=f"x{i}_{j}")
    y = [model.addVar(vtype="B", name=f"y{i}") for i in range(n)]
    
    for i in range(n):
        model.addCons(
            quicksum(x[i, j] for j in range(n)) == 1
        )
        
    for j in range(n):
        model.addCons(
            quicksum(sizes[i] * x[i, j] for i in range(n)) <= capacity * y[j]
        )
        
    model.setObjective(
        quicksum(y[j] for j in range(n)), "minimize"
    )
               
    return model