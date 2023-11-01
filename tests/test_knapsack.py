from pyscipopt import Model, quicksum

def test_knapsack():
    # create solver instance
    s = Model("Knapsack")
    s.hideOutput()

    # setting the objective sense to maximise
    s.setMaximize()

    # item weights
    weights = [4, 2, 6, 3, 7, 5]
    # item costs
    costs = [7, 2, 5, 4, 3, 4]

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

    # solve problem
    s.optimize()

    s.printStatistics()

    # print solution
    varSolutions = []
    for i in range(len(weights)):
        solValue = round(s.getVal(knapsackVars[i]))
        varSolutions.append(solValue)
        if solValue > 0:
            print (varNames[i], "Times Selected:", solValue)
            print ("\tIncluded Weight:", weights[i]*solValue, "\tItem Cost:", costs[i]*solValue)

    includedWeight = sum([weights[i]*varSolutions[i] for i in range(len(weights))])
    assert includedWeight > 0 and includedWeight <= knapsackSize