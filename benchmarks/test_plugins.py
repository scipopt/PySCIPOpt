from pyscipopt import SCIP_EVENTTYPE, SCIP_RESULT, Branchrule, Conshdlr, Eventhdlr, Model


class AllFixed(Conshdlr):
    def __init__(self, vars):
        self.vars = vars

    def _result(self):
        if all(v.getLbLocal() == v.getUbLocal() for v in self.vars):
            return {"result": SCIP_RESULT.CUTOFF}
        return {"result": SCIP_RESULT.INFEASIBLE}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        return {"result": SCIP_RESULT.FEASIBLE}

    def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
        return self._result()

    def consenfolp(self, constraints, nusefulconss, solinfeasible):
        return self._result()

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        for v in self.vars:
            self.model.addVarLocksType(v, locktype, nlockspos + nlocksneg, nlockspos + nlocksneg)


class FirstFree(Branchrule):
    def branchexecps(self, allowaddcons):
        cands, *_ = self.model.getPseudoBranchCands()
        self.model.branchVar(cands[0])
        return {"result": SCIP_RESULT.BRANCHED}


class NodeEvents(Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        event.getType()
        node = self.model.getCurrentNode()
        node.getNumber()
        node.getDepth()
        self.model.getNNodes()


def _plugin_model(n=30, nodes=20_000):
    m = Model()
    m.hideOutput()
    m.setPresolve(0)
    m.setHeuristics(3)
    m.setSeparating(3)
    m.setParam("lp/solvefreq", -1)
    m.setParam("limits/nodes", nodes)
    x = [m.addVar(vtype="B") for _ in range(n)]
    m.includeConshdlr(AllFixed(x), "allfixed", "", chckpriority=-1, enfopriority=-1, needscons=False)
    m.includeBranchrule(FirstFree(), "firstfree", "", priority=1000000, maxdepth=-1, maxbounddist=1)
    m.includeEventhdlr(NodeEvents(), "nodeevents", "")
    return m


def test_plugin_callbacks(bench):
    bench(lambda m: m.optimize(), setup=lambda: ((_plugin_model(),), {}), rounds=3)
