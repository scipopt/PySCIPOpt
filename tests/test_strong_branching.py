from pyscipopt import Model, Branchrule, SCIP_RESULT, quicksum, SCIP_PARAMSETTING

"""
This is a test for strong branching. It also gives a basic outline of what a function that imitates strong
branching should do. An example of when to use this would be to access strong branching scores etc. 

If this was done fully, then one would need to handle the following cases:
- What happens if a new primal solution is found and the bound is larger than the cutoff bound? Return CUTOFF status
- What happens if the bound for one of the children is above a cutoff bound? Change variable bound
- If probing is ever enabled then one would need to handle new found bounds appropriately.
"""


class StrongBranchingRule(Branchrule):

    def __init__(self, scip, idempotent=False):
        self.scip = scip
        self.idempotent = idempotent

    def branchexeclp(self, allowaddcons):

        # Get the branching candidates. Only consider the number of priority candidates (they are sorted to be first)
        # The implicit integer candidates in general shouldn't be branched on. Unless specified by the user
        # npriocands and ncands are the same (npriocands are variables that have been designated as priorities)
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

        # Initialise scores for each variable
        scores = [-self.scip.infinity() for _ in range(npriocands)]
        down_bounds = [None for _ in range(npriocands)]
        up_bounds = [None for _ in range(npriocands)]

        # Initialise placeholder values
        num_nodes = self.scip.getNNodes()
        lpobjval = self.scip.getLPObjVal()
        lperror = False
        best_cand_idx = 0

        # Start strong branching and iterate over the branching candidates
        self.scip.startStrongBranching()
        for i in range(npriocands):

            # Check the case that the variable has already been strong branched on at this node.
            # This case occurs when events happen in the node that should be handled immediately.
            # When processing the node again (because the event did not remove it), there's no need to duplicate work.
            if self.scip.getVarStrongBranchLastNode(branch_cands[i]) == num_nodes:
                down, up, downvalid, upvalid, _, lastlpobjval = self.scip.getVarStrongBranchInfoLast(branch_cands[i])
                if downvalid:
                    down_bounds[i] = down
                if upvalid:
                    up_bounds[i] = up
                downgain = max([down - lastlpobjval, 0])
                upgain = max([up - lastlpobjval, 0])
                scores[i] = self.scip.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])
                continue

            # Strong branch!
            down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, lperror = self.scip.strongBranchVar(
                branch_cands[i], 200, idempotent=self.idempotent)

            # In the case of an LP error handle appropriately (for this example we just break the loop)
            if lperror:
                break

            # In the case of both infeasible sub-problems cutoff the node
            if not self.idempotent and downinf and upinf:
                return {"result": SCIP_RESULT.CUTOFF}

            # Calculate the gains for each up and down node that strong branching explored
            if not downinf and downvalid:
                down_bounds[i] = down
                downgain = max([down - lpobjval, 0])
            else:
                downgain = 0
            if not upinf and upvalid:
                up_bounds[i] = up
                upgain = max([down - lpobjval, 0])
            else:
                upgain = 0

            # Update the pseudo-costs
            if not self.idempotent:
                lpsol = branch_cands[i].getLPSol()
                if not downinf and downvalid:
                    self.scip.updateVarPseudoCost(branch_cands[i], -self.scip.frac(lpsol), downgain, 1)
                if not upinf and upvalid:
                    self.scip.updateVarPseudoCost(branch_cands[i], 1 - self.scip.frac(lpsol), upgain, 1)

            scores[i] = self.scip.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])
            if scores[i] > scores[best_cand_idx]:
                best_cand_idx = i

        # End strong branching
        self.scip.endStrongBranching()

        # In the case of an LP error
        if lperror:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Branch on the variable with the largest score
        down_child, eq_child, up_child = self.model.branchVarVal(
            branch_cands[best_cand_idx], branch_cands[best_cand_idx].getLPSol())

        # Update the bounds of the down node and up node
        if self.scip.allColsInLP() and not self.idempotent:
            if down_child is not None and down_bounds[best_cand_idx] is not None:
                self.scip.updateNodeLowerbound(down_child, down_bounds[best_cand_idx])
            if up_child is not None and up_bounds[best_cand_idx] is not None:
                self.scip.updateNodeLowerbound(up_child, up_bounds[best_cand_idx])

        return {"result": SCIP_RESULT.BRANCHED}


class FeatureSelectorBranchingRule(Branchrule):

    def __init__(self, scip):
        self.scip = scip

    def branchexeclp(self, allowaddcons):

        if self.scip.getNNodes() == 1 or self.scip.getNNodes() == 250:

            rows = self.scip.getLPRowsData()
            cols = self.scip.getLPColsData()

            # This is just a dummy rule to check functionality.
            # A user should be able to see how they can access information without affecting the solve process.

            age_row = rows[0].getAge()
            age_col = cols[0].getAge()
            red_cost_col = self.scip.getColRedCost(cols[0])

            avg_sol = cols[0].getVar().getAvgSol()

        return {"result": SCIP_RESULT.DIDNOTRUN}


def create_model():
    scip = Model()
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    # scip.setIntParam("presolving/maxrounds", 0)
    scip.setLongintParam("limits/nodes", 2000)

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

    return scip


def test_strong_branching():
    scip = create_model()

    strong_branch_rule = StrongBranchingRule(scip, idempotent=False)
    scip.includeBranchrule(strong_branch_rule, "strong branch rule", "custom strong branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)

    scip.optimize()


def test_strong_branching_idempotent():
    scip = create_model()

    strong_branch_rule = StrongBranchingRule(scip, idempotent=True)
    scip.includeBranchrule(strong_branch_rule, "strong branch rule", "custom strong branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)

    scip.optimize()


def test_dummy_feature_selector():
    scip = create_model()

    feature_dummy_branch_rule = FeatureSelectorBranchingRule(scip)
    scip.includeBranchrule(feature_dummy_branch_rule, "dummy branch rule", "custom feature creation branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)

    scip.optimize()