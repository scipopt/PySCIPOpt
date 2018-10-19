from pyscipopt import Model, Branchrule, SCIP_RESULT


class MyBranching(Branchrule):

    def __init__(self, model, cont, integral):
        self.model = model
        self.cont = cont
        self.integral = integral
        self.count = 0
        self.was_called_val = False
        self.was_called_int = False

    def branchexeclp(self, allowaddcons):
        print("in branchexelp")
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

        if self.count == 1:
            down, eq, up = self.model.branchVarVal(self.cont, 1.3)
            self.model.chgVarLbNode(down, self.cont, -1.5)
            self.model.chgVarUbNode(up, self.cont, 3.0)
            self.was_called_val = True
            down2, eq2, up2 = self.model.branchVar(self.integral)
            self.was_called_int = True
            self.model.createChild(6, 7)
            return {"result": SCIP_RESULT.BRANCHED}




m = Model()

x0 = m.addVar(lb=-2, ub=4)
r1 = m.addVar()
r2 = m.addVar()
y0 = m.addVar(lb=3)
t = m.addVar(lb=None)
l = m.addVar(vtype="I", lb=-9, ub=18)
u = m.addVar(vtype="I", lb=-3, ub=99)



m.addCons(r1 >= x0)
m.addCons(r2 >= -x0)
m.addCons(y0 == r1 +r2)
m.addCons(t * l + l * u >= 4)

m.setObjective(t)
m.addCons(t >= r1 * (r1 - x0) + r2 * (r2 + x0))

my_branchrule = MyBranching(m, x0, l)
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

