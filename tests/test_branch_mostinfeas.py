from pyscipopt import Branchrule, SCIP_RESULT
from helpers.utils import random_mip_1


class MostInfBranchRule(Branchrule):

    def __init__(self, scip):
        self.scip = scip

    def branchexeclp(self, allowaddcons):

        # Get the branching candidates. Only consider the number of priority candidates (they are sorted to be first)
        # The implicit integer candidates in general shouldn't be branched on. Unless specified by the user
        # npriocands and ncands are the same (npriocands are variables that have been designated as priorities)
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

        # Find the variable that is most fractional
        best_cand_idx = 0
        best_dist = float('inf')
        for i in range(npriocands):
            if abs(branch_cand_fracs[i] - 0.5) <= best_dist:
                best_dist = abs(branch_cand_fracs[i] - 0.5)
                best_cand_idx = i

        # Branch on the variable with the largest score
        down_child, eq_child, up_child = self.model.branchVarVal(
            branch_cands[best_cand_idx], branch_cand_sols[best_cand_idx])

        return {"result": SCIP_RESULT.BRANCHED}


def test_branch_mostinfeas():
    scip = random_mip_1(node_lim=1000, small=True)
    most_inf_branch_rule = MostInfBranchRule(scip)
    scip.includeBranchrule(most_inf_branch_rule, "python-mostinf", "custom most infeasible branching rule",
                           priority=10000000, maxdepth=-1, maxbounddist=1)
    scip.optimize()
