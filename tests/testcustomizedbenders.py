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
        self.strong = {}
        self.name = name  # benders name

    def benderscreatesub(self, probnumber):
        subprob = Model("flp-subprob")
        x, y = {}, {}
        for j in self.J:
            y[j] = subprob.addVar(vtype="B", name="y(%s)" % j)
            for i in self.I:
                x[i, j] = subprob.addVar(vtype="C", name="x(%s,%s)" % (i, j))
        for i in self.I:
            self.demand[i] = subprob.addCons(quicksum(x[i, j] for j in self.J) == self.d[i], "Demand(%s)" % i)

        for j in self.M:
            self.capacity[j] = subprob.addCons(quicksum(x[i, j] for i in self.I) <= self.M[j] * y[j], "Capacity(%s)" % i)

        for (i, j) in x:
            self.strong[i,j] = subprob.addCons(x[i, j] <= self.d[i] * y[j], "Strong(%s,%s)" % (i, j))

        subprob.setObjective(
            quicksum(self.c[i, j] * x[i, j] for i in self.I for j in self.J),
            "minimize")
        subprob.data = x, y
        self.model.addBendersSubproblem(self.name, subprob)
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
        print("Solving subproblem:", probnumber)
        SCIP_BENDERSENFOTYPE_LP = 1
        SCIP_BENDERSENFOTYPE_RELAX = 2
        SCIP_BENDERSENFOTYPE_PSEUDO = 3
        SCIP_BENDERSENFOTYPE_CHECK = 4

        self.model.setupBendersSubproblem(probnumber, self, solution)
        self.subprob.solveProbingLP()
        subprob = self.model.getBendersSubproblem(probnumber, self)
        assert self.subprob.getObjVal() == subprob.getObjVal()

        subprob.printSol()
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

        print("Result:", result)

        return result_dict


    def bendersfreesub(self, probnumber):
        print("Freeing subproblems")
        if self.subprob.inProbing():
           self.subprob.endProbing()

class testBenderscut(Benderscut):

   def __init__(self, I, J, M):
      self.I, self.J, self.M = I, J, M

   def benderscutexec(self, solution, probnumber, enfotype):
      print("Generating Benders cuts")
      subprob = self.model.getBendersSubproblem(probnumber, benders=self.benders)
      membersubprob = self.benders.subprob

      # testing whether the dual multipliers can be found for the retrieved
      # subproblem model. If the constraints don't exist, then the subproblem
      # model is not correct.
      # Also checking whether the dual multiplier is the same between the
      # member subproblem and the retrieved subproblem
      for i in self.I:
         subprobcons = self.benders.demand[i]
         try:
            dualmult = subprob.getDualMultiplier(subprobcons)
         except:
            print("Subproblem constraint <%d> does not exist in the "\
                  "subproblem."%subprobcons.name)
            assert False

         memberdualmult = membersubprob.getDualMultiplier(subprobcons)
         if dualmult != memberdualmult:
            print("The dual multipliers between the two subproblems are not "\
                  "the same.")
            assert False

      return {"result" : SCIP_RESULT.FEASIBLE}



def flp(J,f):
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

    master.setObjective(
        quicksum(f[j]*y["y(%d)"%j] for j in J),
        "minimize")
    master.data = y

    return master


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
    master = flp(J, f)
    # initializing the default Benders' decomposition with the subproblem
    master.setPresolve(SCIP_PARAMSETTING.OFF)
    master.setBoolParam("misc/allowdualreds", False)
    master.setBoolParam("benders/copybenders", False)
    bendersName = "testBenders"
    benderscutName = "testBenderscut"
    testbd = testBenders(master.data, I, J, M, c, d, bendersName)
    testbdc = testBenderscut(I, J, M)
    master.includeBenders(testbd, bendersName, "benders plugin")
    #master.includeBenderscut(testbd, testbdc, benderscutName,
          #"benderscut plugin", priority=1000000)
    master.activateBenders(bendersName, 1)
    master.setBoolParam("constraints/benders/active", True)
    master.setBoolParam("constraints/benderslp/active", True)
    # optimizing the problem using Benders' decomposition
    master.optimize()

    # solving the subproblems to get the best solution
    # master.computeBestSolSubproblems()

    EPS = 1.e-6
    y = master.data
    facilities = [j for j in y if master.getVal(y[j]) > EPS]

    x, suby = testbd.subprob.data
    edges = [(i, j) for (i, j) in x if testbd.subprob.getVal(x[i,j]) > EPS]

    print("Optimal value:", master.getObjVal())
    print("Facilities at nodes:", facilities)
    print("Edges:", edges)

    master.printStatistics()

    # since computeBestSolSubproblems() was called above, we need to free the
    # subproblems. This must happen after the solution is extracted, otherwise
    # the solution will be lost
    master.freeBendersSubproblems()

    assert master.getObjVal() == 5.61e+03


if __name__ == "__main__":
    test_flpbenders()
