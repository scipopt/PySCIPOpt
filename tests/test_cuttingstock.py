import pyscipopt.scip as scip

# The reduced cost function for the variable pricer
def py_scip_redcost(solver, pricer):
    pricerdata = pricer.getPricerData()

    # Retreiving the dual solutions
    dualSolutions = []
    for i, c in enumerate(pricerdata.cons):
        dualSolutions.append(solver.getDualsolLinear(c))

    # Building a MIP to solve the subproblem
    subMIP = scip.Model("CuttingStock-Sub")

    # Turning off presolve
    subMIP.setPresolve(scip.scip_paramsetting.off)

    # Setting the verbosity level to 0
    subMIP.hideOutput()

    cutWidthVars = []
    varNames = []
    varBaseName = "CutWidth"

    # Variables for the subMIP
    for i in range(len(dualSolutions)):
        varNames.append(varBaseName + "_" + str(i))
        cutWidthVars.append(subMIP.addVar(varNames[i], vtype = "I", obj = -1.0 * dualSolutions[i]))

    # Adding the knapsack constraint
    knapsackCoeffs = {cutWidthVars[i] : pricerdata.widths[i] for i in range(len(pricerdata.widths))}
    knapsackCons = subMIP.addCons(knapsackCoeffs, lhs = None, rhs = pricerdata.rollLength)

    # Solving the subMIP to generate the most negative reduced cost pattern
    subMIP.optimize()

    objval = 1 + subMIP.getObjVal()

    # Adding the column to the master problem
    if objval < -1e-08:
        currentNumVar = len(pricerdata.var)

        # Creating new var; must set pricedVar to True
        newVar = solver.addVar("NewPattern_" + str(currentNumVar), vtype = "C", obj = 1.0, pricedVar = True)

        # Adding the new variable to the constraints of the master problem
        newPattern = []
        for i, c in enumerate(pricerdata.cons):
            coeff = round(subMIP.getVal(cutWidthVars[i]))
            solver.addConsCoeff(c, newVar, coeff)

            newPattern.append(coeff)

        # Storing the new variable in the pricer data.
        pricerdata.patterns.append(newPattern)
        pricerdata.var.append(newVar)

    # Freeing the subMIP
    subMIP.free()

    return scip.scip_result.success


# The initialisation function for the variable pricer
def py_scip_init(solver, pricer):
    pricerdata = pricer.getPricerData()
    transformProbCons(solver, pricerdata.cons, pricerdata)


# A user defined function to retrieve the transformed constraints of the problem
def transformProbCons(solver, cons, pricerdata):
    for i, c in enumerate(cons):
        pricerdata.cons[i] = solver.getTransformedCons(c.cons)


def test_cuttingstock():
    # create solver instance
    s = scip.Model("CuttingStock")

    s.setPresolve(0)

    # creating a pricer
    pricer = scip.Pricer()
    s.includePricer(pricer, "CuttingStockPricer", "Pricer to identify new cutting stock patterns")
    scip.py_pricerdata.name = "Testing"

    # this links the user defined reduced cost function to the functions defined with cython
    scip.py_scip_redcost = py_scip_redcost
    scip.py_scip_init = py_scip_init

    # item widths
    widths = [14, 31, 36, 45]
    # width demand
    demand = [211, 395, 610, 97]
    # roll length
    rollLength = 100
    assert len(widths) == len(demand)

    # adding the initial variables
    cutPatternVars = []
    varNames = []
    varBaseName = "Pattern"
    patterns = []

    initialCoeffs = []
    for i in range(len(widths)):
        varNames.append(varBaseName + "_" + str(i))
        cutPatternVars.append(s.addVar(varNames[i], obj = 1.0))

    # adding a linear constraint for the knapsack constraint
    demandCons = []
    for i in range(len(widths)):
        numWidthsPerRoll = float(int(rollLength/widths[i]))
        coeffs = {cutPatternVars[i] : numWidthsPerRoll}
        demandCons.append(s.addCons(coeffs, lhs = demand[i], separate = False, modifiable = True))
        newPattern = [0]*len(widths)
        newPattern[i] = numWidthsPerRoll
        patterns.append(newPattern)

    # Setting the pricer_data for use in the init and redcost functions
    scip.py_pricerdata.var = cutPatternVars
    scip.py_pricerdata.cons = demandCons
    scip.py_pricerdata.transcons = []
    scip.py_pricerdata.widths = widths
    scip.py_pricerdata.demand = demand
    scip.py_pricerdata.rollLength = rollLength
    scip.py_pricerdata.patterns = patterns

    # solve problem
    s.optimize()

    # print original data
    printWidths = '\t'.join(str(e) for e in widths)
    print('\nInput Data')
    print('==========')
    print('Roll Length:', rollLength)
    print('Widths:\t', printWidths)
    print('Demand:\t', '\t'.join(str(e) for e in demand))

    # print solution
    widthOutput = [0]*len(widths)
    print('\nResult')
    print('======')
    print('\t\tSol Value', '\tWidths\t', printWidths)
    for i in range(len(scip.py_pricerdata.var)):
        rollUsage = 0
        solValue = round(s.getVal(scip.py_pricerdata.var[i]))
        if solValue > 0:
            print('Pattern_' + str(i) + ':\t', solValue, '\t\tCuts:\t', end=' ')
            for j in range(len(widths)):
                rollUsage += scip.py_pricerdata.patterns[i][j]*widths[j]
                widthOutput[j] += scip.py_pricerdata.patterns[i][j]*solValue
                print(scip.py_pricerdata.patterns[i][j], '\t', end=' ')
            print('Usage:', rollUsage)

    print('\t\t\tTotal Output:','\t'.join(str(e) for e in widthOutput))


if __name__ == '__main__':
    test_cuttingstock()
