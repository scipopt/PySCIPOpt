import pytest
from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum

class CutPricer(Pricer):

    # The reduced cost function for the variable pricer
    def pricerredcost(self):

        # Retrieving the dual solutions
        dualSolutions = []
        for i, c in enumerate(self.data['cons']):
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
        knapsackCons = subMIP.addCons(
            quicksum(w*v for (w,v) in zip(self.data['widths'], cutWidthVars)) <= self.data['rollLength'])

        # Solving the subMIP to generate the most negative reduced cost pattern
        subMIP.optimize()

        objval = 1 + subMIP.getObjVal()

        # testing methods
        assert type(self.model.getNSolsFound()) == int
        assert type(self.model.getNBestSolsFound()) == int
        assert self.model.getNBestSolsFound() <= self.model.getNSolsFound()
        
        self.model.data["nSols"] = self.model.getNSolsFound()

        # Adding the column to the master problem (model.LT because of numerics)
        if self.model.isLT(objval, 0): 
            currentNumVar = len(self.data['var'])

            # Creating new var; must set pricedVar to True
            newVar = self.model.addVar("NewPattern_" + str(currentNumVar), vtype = "C", obj = 1.0, pricedVar = True)

            # Adding the new variable to the constraints of the master problem
            newPattern = []
            for i, c in enumerate(self.data['cons']):
                coeff = round(subMIP.getVal(cutWidthVars[i]))
                self.model.addConsCoeff(c, newVar, coeff)

                newPattern.append(coeff)

            # Testing getVarRedcost
            assert self.model.isEQ(self.model.getVarRedcost(newVar), objval)

            # Storing the new variable in the pricer data.
            self.data['patterns'].append(newPattern)
            self.data['var'].append(newVar)

        return {'result':SCIP_RESULT.SUCCESS}

    # The initialisation function for the variable pricer to retrieve the transformed constraints of the problem
    def pricerinit(self):
        for i, c in enumerate(self.data['cons']):
            self.data['cons'][i] = self.model.getTransformedCons(c)


def test_cuttingstock():
    # create solver instance
    s = Model("CuttingStock")

    s.setPresolve(0)
    s.data = {}
    s.data["nSols"] = 0

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

    for i in range(len(widths)):
        varNames.append(varBaseName + "_" + str(i))
        cutPatternVars.append(s.addVar(varNames[i], obj = 1.0))

    # adding a linear constraint for the knapsack constraint
    demandCons = []
    for i in range(len(widths)):
        numWidthsPerRoll = float(int(rollLength/widths[i]))
        demandCons.append(s.addCons(numWidthsPerRoll*cutPatternVars[i] >= demand[i],
                                    separate = False, modifiable = True))
        newPattern = [0]*len(widths)
        newPattern[i] = numWidthsPerRoll
        patterns.append(newPattern)

    # Setting the pricer_data for use in the init and redcost functions
    pricer.data = {}
    pricer.data['var'] = cutPatternVars
    pricer.data['cons'] = demandCons
    pricer.data['widths'] = widths
    pricer.data['demand'] = demand
    pricer.data['rollLength'] = rollLength
    pricer.data['patterns'] = patterns
    pricer.data['redcosts'] = []

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
    for i in range(len(pricer.data['var'])):
        rollUsage = 0
        solValue = s.getVal(pricer.data['var'][i])
        if s.isGT(solValue, 0):
            outline = 'Pattern_' + str(i) + ':\t' + str(solValue) + '\t\tCuts:\t '
            for j in range(len(widths)):
                rollUsage += pricer.data['patterns'][i][j]*widths[j]
                widthOutput[j] += pricer.data['patterns'][i][j]*solValue
                outline += str(pricer.data['patterns'][i][j]) + '\t'
            outline += 'Usage:' + str(rollUsage)
            print(outline)

    print('\t\t\tTotal Output:\t', '\t'.join(str(e) for e in widthOutput))
    
    assert s.getObjVal() == 452.25
    assert type(s.getNSols()) == int
    assert s.getNSols() == s.data["nSols"]

    # Testing freeTransform
    s.freeTransform()
    for i in range(10):
        s.addVar()

def test_incomplete_pricer():
    class IncompletePricer(Pricer):
        pass

    pricer = IncompletePricer()
    model = Model()
    model.setPresolve(0)    
    model.includePricer(pricer, "", "") 

    with pytest.raises(Exception):
        model.optimize()