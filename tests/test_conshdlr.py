from pyscipopt import Model, Conshdlr, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
from types import SimpleNamespace

class MyConshdlr(Conshdlr):
    def createData(self, constraint, nvars, othername):
        print("Creating data for my constraint: %s"%constraint.name)
        constraint.data = SimpleNamespace()
        constraint.data._nvars = nvars
        constraint.data._myothername = othername


    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        print("consenforcinglp python style")
        for cons in constraints:
            if cons.data._nvars > 0:
                result = SCIP_RESULT.FEASIBLE
                print(cons.data._myothername, "is feasible while enforcing")
        return {"result": result}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason):
        print("conscheking python style")
        for cons in constraints:
            if cons.data._nvars > 0:
                result = SCIP_RESULT.FEASIBLE
                print(cons.data._myothername, "is so feasible...")
        return {"result": result}

    def conslock(self, constraints, nlockspos, nlocksneg):
        print("don't care about locks.. don't even have a variable!")
        pass


def test_conshdlr():
    # create solver instance
    s = Model()

    # create conshdlr and include it to SCIP
    conshdlr = MyConshdlr()
    s.includeConshdlr(conshdlr, "PyCons", "custom constraint handler implemented in python",
                          sepapriority=0, enfopriority=0, chckpriority=0, sepafreq=1, propfreq=-1,
                          eagerfreq=-1, maxprerounds=0, delaysepa=False, delayprop=False, needscons=True, 
                          presoltiming=SCIP_PRESOLTIMING.FAST, proptiming=SCIP_PROPTIMING.BEFORELP)

    cons1 = s.createCons(conshdlr, "cons1name")
    cons2 = s.createCons(conshdlr, "cons2name")
    conshdlr.createData(cons1, 10, "cons1_anothername")
    conshdlr.createData(cons2, 12, "cons2_anothername")

    # add these constraints
    s.addPyCons(cons1)
    s.addPyCons(cons2)
    print("constraints have been added!")

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

    #s.printStatistics()

if __name__ == "__main__":
    test_conshdlr()
