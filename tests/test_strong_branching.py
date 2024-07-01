from pyscipopt import Model, Branchrule, SCIP_RESULT, quicksum, SCIP_PARAMSETTING

import pdb

class StrongBranchingRule(Branchrule):

    def __init__(self, scip):
        self.scip = scip
        self.count = 0

    def branchexeclp(self, allowaddcons):

        # Get the branching candidates. Only consider the number of priority candidates (they are sorted to be first)
        # The implicit integer candidates in general shouldn't be branched on. Unless specified by the user
        # npriocands and ncands are the same (npriocands are variables that have been designated as priorities)
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

        # Initialise scores for each variable
        scores = [0 for _ in range(npriocands)]

        # Start strong branching and iterate over the branching candidates
        num_nodes = self.scip.getNNodes()
        self.scip.startStrongBranching(False)
        for i in range(npriocands):

            if self.scip.getVarStrongBranchNode(branch_cands[i]) == num_nodes:
                down, up, _, _, _, lastlpobjval = self.scip.getVarStrongBranchLast(branch_cands[i])
                downgain = max([down - lastlpobjval, 0])
                upgain = max([up - lastlpobjval, 0])
                scores[i] = self.scip.getBranchScoreMultiple([downgain, upgain])


            down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, lperror = self.scip.getVarStrongBranch(branch_cands[i], 200)

            # In the case of an LP error handle appropriately (for this example we ignore the score)
            if lperror:
                continue
            elif not downvalid:




        pdb.set_trace()



def test_strong_branching():

    scip = Model()
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    # scip.setIntParam("presolving/maxrounds", 0)
    scip.setLongintParam("limits/nodes", 5)

    x0 = scip.addVar(lb=-2, ub=4)
    r1 = scip.addVar()
    r2 = scip.addVar()
    y0 = scip.addVar(lb=3)
    t = scip.addVar(lb=None)
    l = scip.addVar(vtype="I", lb=-9, ub=18)
    u = scip.addVar(vtype="I", lb=-3, ub=99)

    more_vars = []
    for i in range(100):
        more_vars.append(scip.addVar(vtype="I", lb=-12, ub=40))
        scip.addCons(quicksum(v for v in more_vars) <= (40 - i) * quicksum(v for v in more_vars[::2]))

    for i in range(100):
        more_vars.append(scip.addVar(vtype="I", lb=-52, ub=10))
        scip.addCons(quicksum(v for v in more_vars[50::2]) <= (40 - i) * quicksum(v for v in more_vars[405::2]))

    scip.addCons(r1 >= x0)
    scip.addCons(r2 >= -x0)
    scip.addCons(y0 == r1 + r2)
    scip.addCons(t + l + 7 * u <= 300)
    scip.addCons(t >= quicksum(v for v in more_vars[::3]) - 10 * more_vars[5] + 5 * more_vars[9])
    scip.addCons(more_vars[3] >= l + 2)
    scip.addCons(7 <= quicksum(v for v in more_vars[::4]) - x0)
    scip.addCons(quicksum(v for v in more_vars[::2]) + l <= quicksum(v for v in more_vars[::4]))

    scip.setObjective(t - quicksum(j * v for j, v in enumerate(more_vars[20:-40])))

    strong_branch_rule = StrongBranchingRule(scip)
    scip.includeBranchrule(strong_branch_rule, "strong branch rule", "custom strong branching rule",
                        priority=10000000, maxdepth=4, maxbounddist=1)

    scip.optimize()

test_strong_branching()