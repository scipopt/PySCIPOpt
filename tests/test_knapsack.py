import pyscipopt.scip as scip

def test_knapsack():
    # create solver instance
    s = scip.Solver()
    s.create()
    s.includeDefaultPlugins()
    s.createProbBasic("Knapsack")

    # setting the objective sense to maximise
    s.setMaximise()

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
        knapsackVars.append(s.addIntVar(varNames[i], obj=costs[i], ub=1.0))


    # adding a linear constraint for the knapsack constraint
    coeffs = {knapsackVars[i]: weights[i] for i in range(len(weights))}
    s.addCons(coeffs, lhs=None, rhs=knapsackSize)

    # solve problem
    s.solve()

    s.printStatistics()

    # retrieving the best solution
    solution = s.getBestSol()

    # print solution
    print()
    varSolutions = []
    for i in range(len(weights)):
        solValue = round(s.getVal(solution, knapsackVars[i]))
        varSolutions.append(solValue)
        if solValue > 0:
            print (varNames[i], "Times Selected:", solValue)
            print ("\tIncluded Weight:", weights[i]*solValue, "\tItem Cost:", costs[i]*solValue)

        s.releaseVar(knapsackVars[i])



    includedWeight = sum([weights[i]*varSolutions[i] for i in range(len(weights))])
    assert includedWeight > 0 and includedWeight <= knapsackSize

    s.free()

if __name__ == '__main__':
    test_knapsack()
