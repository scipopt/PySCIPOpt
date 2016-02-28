from pyscipopt import Model, Heur, SCIP_RESULT, SCIP_PARAMSETTING


class MyHeur(Heur):

    def heurexec(self, heurtiming, nodeinfeasible):

        sol = self.model.createSol(self)
        vars = self.model.getVars()

        self.model.setSolVal(sol, vars[0], 5.0)
        self.model.setSolVal(sol, vars[1], 0.0)

        print(vars[0])

        accepted = self.model.trySol(sol)

        if accepted:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}


def test_heur():
    # create solver instance
    s = Model()
    heuristic = MyHeur()
    s.includeHeur(heuristic, "PyHeur", "custom heuristic implemented in python", "Y")
    s.setPresolve(SCIP_PARAMSETTING.OFF)

    # add some variables
    x = s.addVar("x", obj=1.0)
    y = s.addVar("y", obj=2.0)

    # add some constraint
    s.addCons(x + 2*y >= 5)

    # solve problem
    s.optimize()

    # print solution
    assert round(s.getVal(x)) == 5.0
    assert round(s.getVal(y)) == 0.0

    s.printStatistics()

if __name__ == "__main__":
    test_heur()
