from pyscipopt import Model, Branchrule, SCIP_RESULT, quicksum, SCIP_PARAMSETTING


class MyBranching(Branchrule):

    def __init__(self, model, cont):
        self.model = model
        self.cont = cont
        self.count = 0
        self.was_called_val = False
        self.was_called_int = False

    def branchexeclp(self, allowaddcons):
        self.count += 1
        if self.count >= 2:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        assert allowaddcons

        assert not self.model.inRepropagation()
        assert not self.model.inProbing()
        self.model.startProbing()
        assert not self.model.isObjChangedProbing()
        self.model.fixVarProbing(self.cont, 2.0)
        self.model.constructLP()
        self.model.solveProbingLP()
        self.model.getLPObjVal()
        self.model.endProbing()

        self.integral = self.model.getLPBranchCands()[0][0]

        if self.count == 1:
            down, eq, up = self.model.branchVarVal(self.cont, 1.3)
            self.model.chgVarLbNode(down, self.cont, -1.5)
            self.model.chgVarUbNode(up, self.cont, 3.0)
            self.was_called_val = True
            down2, eq2, up2 = self.model.branchVar(self.integral)
            self.was_called_int = True
            self.model.createChild(6, 7)
            return {"result": SCIP_RESULT.BRANCHED}


def test_branching():
    m = Model()
    m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.setIntParam("presolving/maxrounds", 0)
    #m.setLongintParam("lp/rootiterlim", 3)
    m.setLongintParam("limits/nodes", 1)

    x0 = m.addVar(lb=-2, ub=4)
    r1 = m.addVar()
    r2 = m.addVar()
    y0 = m.addVar(lb=3)
    t = m.addVar(lb=None)
    l = m.addVar(vtype="I", lb=-9, ub=18)
    u = m.addVar(vtype="I", lb=-3, ub=99)

    more_vars = []
    for i in range(1000):
        more_vars.append(m.addVar(vtype="I", lb= -12, ub=40))
        m.addCons(quicksum(v for v in more_vars) <= (40 - i) * quicksum(v for v in more_vars[::2]))

    for i in range(1000):
        more_vars.append(m.addVar(vtype="I", lb= -52, ub=10))
        m.addCons(quicksum(v for v in more_vars[50::2]) <= (40 - i) * quicksum(v for v in more_vars[405::2]))


    m.addCons(r1 >= x0)
    m.addCons(r2 >= -x0)
    m.addCons(y0 == r1 +r2)
    #m.addCons(t * l + l * u >= 4)
    m.addCons(t + l + 7* u <= 300)
    m.addCons(t >= quicksum(v for v in more_vars[::3]) - 10 * more_vars[5] + 5* more_vars[9])
    m.addCons(more_vars[3] >= l + 2)
    m.addCons(7 <= quicksum(v for v in more_vars[::4]) - x0)
    m.addCons(quicksum(v for v in more_vars[::2]) + l <= quicksum(v for v in more_vars[::4]))


    m.setObjective(t - quicksum(j*v for j, v in enumerate(more_vars[20:-40])))
    #m.addCons(t >= r1 * (r1 - x0) + r2 * (r2 + x0))

    my_branchrule = MyBranching(m, x0)
    m.includeBranchrule(my_branchrule, "test branch", "test branching and probing and lp functions",
                        priority=10000000, maxdepth=3, maxbounddist=1)

    m.optimize()

    print("x0", m.getVal(x0))
    print("r1", m.getVal(r1))
    print("r2", m.getVal(r2))
    print("y0", m.getVal(y0))
    print("t", m.getVal(t))

    assert my_branchrule.was_called_val
    assert my_branchrule.was_called_int