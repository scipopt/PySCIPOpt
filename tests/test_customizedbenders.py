"""
flp-benders.py:  model for solving the capacitated facility location problem using Benders' decomposition

minimize the total (weighted) travel cost from n customers
to some facilities with fixed costs and capacities.

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""
from pyscipopt import Model, quicksum, multidict, SCIP_PARAMSETTING, Benders,\
      Benderscut, SCIP_RESULT, SCIP_LPSOLSTAT


class testBenders(Benders):

    def __init__(self, masterVarDict, I, J, M, c, d, name):
        super(testBenders, self).__init__()
        self.mpVardict = masterVarDict
        self.I, self.J, self.M, self.c, self.d = I, J, M, c, d
        self.demand = {}
        self.capacity = {}
        self.name = name  # benders name

    def benderscreatesub(self, probnumber):
        subprob = Model("flp-subprob")
        x, y = {}, {}
        for j in self.J:
            y[j] = subprob.addVar(vtype="B", name="y(%s)" % j)
            for i in self.I:
                x[i, j] = subprob.addVar(vtype="C", name="x(%s,%s)" % (i, j))
        for i in self.I:
            self.demand[i] = subprob.addCons(quicksum(x[i, j] for j in self.J) >= self.d[i], "Demand(%s)" % i)

        for j in self.M:
            self.capacity[j] = subprob.addCons(quicksum(x[i, j] for i in self.I) <= self.M[j] * y[j], "Capacity(%s)" % j)

        subprob.setObjective(
            quicksum(self.c[i, j] * x[i, j] for i in self.I for j in self.J),
            "minimize")
        subprob.data = x, y
        #self.model.addBendersSubproblem(self.name, subprob)
        self.model.addBendersSubproblem(self, subprob)
        self.subprob = subprob

    def bendersgetvar(self, variable, probnumber):
        try:
            if probnumber == -1:  # convert to master variable
                mapvar = self.mpVardict[variable.name]
            else:
                mapvar = self.subprob.data[1][variable.name]
        except KeyError:
            mapvar = None
        return {"mappedvar": mapvar}

    def benderssolvesubconvex(self, solution, probnumber, onlyconvex):
        self.model.setupBendersSubproblem(probnumber, self, solution)
        self.subprob.solveProbingLP()
        subprob = self.model.getBendersSubproblem(probnumber, self)  
        assert self.subprob.getObjVal() == subprob.getObjVal()

        result_dict = {}

        objective = subprob.infinity()
        result = SCIP_RESULT.DIDNOTRUN
        lpsolstat = self.subprob.getLPSolstat()
        if lpsolstat == SCIP_LPSOLSTAT.OPTIMAL:
           objective = self.subprob.getObjVal()
           result = SCIP_RESULT.FEASIBLE
        elif lpsolstat == SCIP_LPSOLSTAT.INFEASIBLE:
           objective = self.subprob.infinity()
           result = SCIP_RESULT.INFEASIBLE
        elif lpsolstat == SCIP_LPSOLSTAT.UNBOUNDEDRAY:
           objective = self.subprob.infinity()
           result = SCIP_RESULT.UNBOUNDED


        result_dict["objective"] = objective
        result_dict["result"] = result

        return result_dict

    def bendersfreesub(self, probnumber):
        if self.subprob.inProbing():
           self.subprob.endProbing()

class testBenderscut(Benderscut):

   def __init__(self, I, J, M, d):
      self.I, self.J, self.M, self.d = I, J, M, d

   def benderscutexec(self, solution, probnumber, enfotype):
      subprob = self.model.getBendersSubproblem(probnumber, benders=self.benders)
      membersubprob = self.benders.subprob

      # checking whether the subproblem is already optimal, i.e. whether a cut
      # needs to be generated
      if self.model.checkBendersSubproblemOptimality(solution, probnumber,
            benders=self.benders):
         return {"result" : SCIP_RESULT.FEASIBLE}

      # testing whether the dual multipliers can be found for the retrieved
      # subproblem model. If the constraints don't exist, then the subproblem
      # model is not correct.
      # Also checking whether the dual multiplier is the same between the
      # member subproblem and the retrieved subproblem`
      lhs = 0
      for i in self.I:
         subprobcons = self.benders.demand[i]
         try:
            dualmult = subprob.getDualsolLinear(subprobcons)
            lhs += dualmult*self.d[i]
         except:
            print("Subproblem constraint <%d> does not exist in the "\
                  "subproblem."%subprobcons.name)
            assert False

         memberdualmult = membersubprob.getDualsolLinear(subprobcons)
         if dualmult != memberdualmult:
            print("The dual multipliers between the two subproblems are not "\
                  "the same.")
            assert False

      coeffs = [subprob.getDualsolLinear(self.benders.capacity[j])*\
            self.M[j] for j in self.J]

      self.model.addCons(self.model.getBendersAuxiliaryVar(probnumber,
         self.benders) -
         quicksum(self.model.getBendersVar(self.benders.subprob.data[1][j],
         self.benders)*coeffs[j] for j in self.J) >= lhs)

      return {"result" : SCIP_RESULT.CONSADDED}



def flp(I, J, M, d,f, c=None, monolithic=False):
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
    # creating the problem
    y = {}
    for j in J:
        y["y(%d)"%j] = master.addVar(vtype="B", name="y(%s)"%j)

    if monolithic:
        x = {}
        demand = {}
        capacity = {}
        for j in J:
            for i in I:
                x[i, j] = master.addVar(vtype="C", name="x(%s,%s)" % (i, j))

        for i in I:
            demand[i] = master.addCons(quicksum(x[i, j] for j in J) >= d[i], "Demand(%s)" % i)

        for j in J:
            print(j, M[j])
            capacity[j] = master.addCons(quicksum(x[i, j] for i in I) <= M[j] * y["y(%d)"%j], "Capacity(%s)" % j)

    master.addCons(quicksum(y["y(%d)"%j]*M[j] for j in J)
          - quicksum(d[i] for i in I) >= 0)

    master.setObjective(
        quicksum(f[j]*y["y(%d)"%j] for j in J) + (0 if not monolithic else
        quicksum(c[i, j] * x[i, j] for i in I for j in J)),
        "minimize")
    master.data = y

    return master


def make_data():
    I,d = multidict({0:80, 1:270, 2:250, 3:160, 4:180})            # demand
    J,M,f = multidict({0:[500,1000], 1:[500,1000], 2:[500,1000]}) # capacity, fixed costs
    c = {(0,0):4,  (0,1):6,  (0,2):9,    # transportation costs
         (1,0):5,  (1,1):4,  (1,2):7,
         (2,0):6,  (2,1):3,  (2,2):4,
         (3,0):8,  (3,1):5,  (3,2):3,
         (4,0):10, (4,1):8,  (4,2):4,
         }
    return I,J,d,M,f,c


def flpbenders_defcuts_test():
    '''
    test the Benders' decomposition plugins with the facility location problem.
    '''
    I,J,d,M,f,c = make_data()
    master = flp(I, J, M, d, f)
    # initializing the default Benders' decomposition with the subproblem
    master.setPresolve(SCIP_PARAMSETTING.OFF)
    master.setBoolParam("misc/allowstrongdualreds", False)
    master.setBoolParam("misc/allowweakdualreds", False)
    master.setBoolParam("benders/copybenders", False)
    bendersName = "testBenders"
    testbd = testBenders(master.data, I, J, M, c, d, bendersName)
    master.includeBenders(testbd, bendersName, "benders plugin")
    master.includeBendersDefaultCuts(testbd)
    master.activateBenders(testbd, 1)
    master.setBoolParam("constraints/benders/active", True)
    master.setBoolParam("constraints/benderslp/active", True)
    master.setBoolParam("benders/testBenders/updateauxvarbound", False)
    # optimizing the problem using Benders' decomposition
    master.optimize()

    # since custom solving functions are defined, we need to manually solve the
    # Benders' decomposition subproblems to get the best solution
    master.setupBendersSubproblem(0, testbd, master.getBestSol())
    testbd.subprob.solveProbingLP()

    EPS = 1.e-6
    y = master.data
    facilities = [j for j in y if master.getVal(y[j]) > EPS]

    x, suby = testbd.subprob.data
    edges = [(i, j) for (i, j) in x if testbd.subprob.getVal(x[i,j]) > EPS]

    print("Optimal value:", master.getObjVal())
    print("Facilities at nodes:", facilities)
    print("Edges:", edges)

    master.printStatistics()

    # since the subproblems were setup and then solved, we need to free the
    # subproblems. This must happen after the solution is extracted, otherwise
    # the solution will be lost
    master.freeBendersSubproblems()

    return master.getObjVal()

def flpbenders_customcuts_test():
    '''
    test the Benders' decomposition plugins with the facility location problem.
    '''
    I,J,d,M,f,c = make_data()
    master = flp(I, J, M, d, f)
    # initializing the default Benders' decomposition with the subproblem
    master.setPresolve(SCIP_PARAMSETTING.OFF)
    master.setBoolParam("misc/allowstrongdualreds", False)
    master.setBoolParam("misc/allowweakdualreds", False)
    master.setBoolParam("benders/copybenders", False)
    bendersName = "testBenders"
    benderscutName = "testBenderscut"
    testbd = testBenders(master.data, I, J, M, c, d, bendersName)
    testbdc = testBenderscut(I, J, M, d)
    master.includeBenders(testbd, bendersName, "benders plugin")
    master.includeBenderscut(testbd, testbdc, benderscutName,
          "benderscut plugin", priority=1000000)
    master.activateBenders(testbd, 1)
    master.setBoolParam("constraints/benders/active", True)
    master.setBoolParam("constraints/benderslp/active", True)
    master.setBoolParam("benders/testBenders/updateauxvarbound", False)
    # optimizing the problem using Benders' decomposition
    master.optimize()

    # since custom solving functions are defined, we need to manually solve the
    # Benders' decomposition subproblems to get the best solution
    master.setupBendersSubproblem(0, testbd, master.getBestSol())
    testbd.subprob.solveProbingLP()

    EPS = 1.e-6
    y = master.data
    facilities = [j for j in y if master.getVal(y[j]) > EPS]

    x, suby = testbd.subprob.data
    edges = [(i, j) for (i, j) in x if testbd.subprob.getVal(x[i,j]) > EPS]

    print("Optimal value:", master.getObjVal())
    print("Facilities at nodes:", facilities)
    print("Edges:", edges)

    master.printStatistics()

    # since the subproblems were setup and then solved, we need to free the
    # subproblems. This must happen after the solution is extracted, otherwise
    # the solution will be lost
    master.freeBendersSubproblems()

    return master.getObjVal()

def flp_test():
    '''
    test the Benders' decomposition plugins with the facility location problem.
    '''
    I,J,d,M,f,c = make_data()
    master = flp(I, J, M, d, f, c=c, monolithic=True)
    # initializing the default Benders' decomposition with the subproblem
    master.setPresolve(SCIP_PARAMSETTING.OFF)

    # optimizing the monolithic problem
    master.optimize()

    EPS = 1.e-6
    y = master.data
    facilities = [j for j in y if master.getVal(y[j]) > EPS]

    print("Optimal value:", master.getObjVal())
    print("Facilities at nodes:", facilities)

    master.printBestSol()
    master.printStatistics()

    return master.getObjVal()


def test_customized_benders():
    defcutsobj = flpbenders_defcuts_test()
    customcutsobj = flpbenders_customcuts_test()
    monolithicobj = flp_test()

    assert defcutsobj == customcutsobj
    assert defcutsobj == monolithicobj
