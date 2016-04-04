from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING

class CutPricer(Pricer):

    # The reduced cost function for the variable pricer
    def pricerredcost(self):

        # Retreiving the dual solutions
        dualSolutions = []
        for i, c in enumerate(self.robert['cons']):
            dualSolutions.append(self.model.getDualsolLinear(c))

        # Building a MIP to solve the subproblem
        subMIP = Model("CuttingStock-Sub")

        # Turning off presolve
        subMIP.setPresolve(SCIP_PARAMSETTING.OFF)

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
        knapsackCoeffs = {cutWidthVars[i] : self.robert['widths'][i] for i in range(len(self.robert['widths']))}
        knapsackCons = subMIP.addCons(knapsackCoeffs, lhs = None, rhs = self.robert['rollLength'])

        # Solving the subMIP to generate the most negative reduced cost pattern
        subMIP.optimize()

        objval = 1 + subMIP.getObjVal()

        # Adding the column to the master problem
        if objval < -1e-08:
            currentNumVar = len(self.robert['var'])

            # Creating new var; must set pricedVar to True
            newVar = self.model.addVar("NewPattern_" + str(currentNumVar), vtype = "C", obj = 1.0, pricedVar = True)

            # Adding the new variable to the constraints of the master problem
            newPattern = []
            for i, c in enumerate(self.robert['cons']):
                coeff = round(subMIP.getVal(cutWidthVars[i]))
                self.model.addConsCoeff(c, newVar, coeff)

                newPattern.append(coeff)

            # Storing the new variable in the pricer data.
            self.robert['patterns'].append(newPattern)
            self.robert['var'].append(newVar)

        # Freeing the subMIP
        subMIP.free()

        return {'result':SCIP_RESULT.SUCCESS}

    # The initialisation function for the variable pricer to retrieve the transformed constraints of the problem
    def pricerinit(self):
        for i, c in enumerate(self.robert['cons']):
            self.robert['cons'][i] = self.model.getTransformedCons(c)


def test_cuttingstock():
    # create solver instance
    s = Model("CuttingStock")

    s.setPresolve(0)

    # creating a pricer
    pricer = CutPricer()
    s.includePricer(pricer, "CuttingStockPricer", "Pricer to identify new cutting stock patterns")

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
    pricer.robert = {}
    pricer.robert['var'] = cutPatternVars
    pricer.robert['cons'] = demandCons
    pricer.robert['widths'] = widths
    pricer.robert['demand'] = demand
    pricer.robert['rollLength'] = rollLength
    pricer.robert['patterns'] = patterns

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
    for i in range(len(pricer.robert['var'])):
        rollUsage = 0
        solValue = round(s.getVal(pricer.robert['var'][i]))
        if solValue > 0:
            outline = 'Pattern_' + str(i) + ':\t' + str(solValue) + '\t\tCuts:\t'
            for j in range(len(widths)):
                rollUsage += pricer.robert['patterns'][i][j]*widths[j]
                widthOutput[j] += pricer.robert['patterns'][i][j]*solValue
                outline += str(pricer.robert['patterns'][i][j]) + '\t'
            outline += 'Usage:' + str(rollUsage)
            print(outline)

    print('\t\t\tTotal Output:','\t'.join(str(e) for e in widthOutput))

    #print('\n')
    #s.printStatistics()

if __name__ == '__main__':
    test_cuttingstock()
