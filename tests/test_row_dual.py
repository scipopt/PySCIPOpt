from pyscipopt import Model, Sepa, SCIP_RESULT, SCIP_PARAMSETTING


class SimpleSepa(Sepa):

    def __init__(self, x, y):
        self.cut = None
        self.x = x
        self.y = y
        self.has_checked = False

    def sepainit(self):
        scip = self.model
        self.trans_x = scip.getTransformedVar(self.x)
        self.trans_y = scip.getTransformedVar(self.y)

    def sepaexeclp(self):
        result = SCIP_RESULT.SEPARATED
        scip = self.model

        if self.cut is not None and not self.has_checked:
            # rhs * dual should be equal to optimal objective (= -1)
            assert scip.isFeasEQ(self.cut.getDualsol(), -1.0)
            self.has_checked = True

        cut = scip.createEmptyRowSepa(self,
                                      lhs=-scip.infinity(),
                                      rhs=1.0)

        scip.cacheRowExtensions(cut)

        scip.addVarToRow(cut, self.trans_x, 1.)
        scip.addVarToRow(cut, self.trans_y, 1.)

        scip.flushRowExtensions(cut)

        scip.addCut(cut, forcecut=True)

        self.cut = cut

        return {"result": result}

    def sepaexit(self):
        assert self.has_checked, "Separator called < 2 times"


def model():
    # create solver instance
    s = Model()

    # turn off presolve
    s.setPresolve(SCIP_PARAMSETTING.OFF)
    # turn off heuristics
    s.setHeuristics(SCIP_PARAMSETTING.OFF)
    # turn off propagation
    s.setIntParam("propagating/maxrounds", 0)
    s.setIntParam("propagating/maxroundsroot", 0)

    # turn off all other separators
    s.setIntParam("separating/strongcg/freq", -1)
    s.setIntParam("separating/gomory/freq", -1)
    s.setIntParam("separating/aggregation/freq", -1)
    s.setIntParam("separating/mcf/freq", -1)
    s.setIntParam("separating/closecuts/freq", -1)
    s.setIntParam("separating/clique/freq", -1)
    s.setIntParam("separating/zerohalf/freq", -1)
    s.setIntParam("separating/mixing/freq", -1)
    s.setIntParam("separating/rapidlearning/freq", -1)
    s.setIntParam("separating/rlt/freq", -1)

    # only two rounds of cuts
    # s.setIntParam("separating/maxroundsroot", 10)

    return s


def test_row_dual():
    s = model()
    # add variable
    x = s.addVar("x", vtype='I', obj=-1, lb=0.)
    y = s.addVar("y", vtype='I', obj=-1, lb=0.)

    # add constraint
    s.addCons(x <= 1.5)
    s.addCons(y <= 1.5)

    # include separator
    sepa = SimpleSepa(x, y)
    s.includeSepa(sepa, "python_simple", "generates a simple cut",
                  priority=1000,
                  freq=1)

    s.addCons(x + y <= 1.75)

    # solve problem
    s.optimize()
