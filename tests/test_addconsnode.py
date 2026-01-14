from pyscipopt import Branchrule, SCIP_RESULT
from helpers.utils import random_mip_1


class MyBranchrule(Branchrule):
    """
    A branching rule that tests addConsNode by adding constraints to child nodes.
    """

    def __init__(self, model):
        self.model = model
        self.addConsNode_called = False
        self.addConsLocal_called = False
        self.branch_var = None

    def branchexeclp(self, allowaddcons):
        if not allowaddcons:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Branch on the first variable (just to test)
        var = self.model.getVars()[0]
        self.branch_var = var

        # Create two child nodes
        child1 = self.model.createChild(1, self.model.getLPObjVal())
        child2 = self.model.createChild(1, self.model.getLPObjVal())

        # Test addConsNode with ExprCons
        cons1 = self.model.addConsNode(child1, var == var.getLbGlobal(), name="branch_down")
        self.addConsNode_called = True
        assert cons1 is not None, "addConsNode should return a Constraint"

        # Making it infeasible to ensure down branch is taken
        cons2 = self.model.addConsNode(child2, var <= var.getUbGlobal()-1, name="branch_up")
        assert cons2 is not None, "addConsNode should return a Constraint"

        return {"result": SCIP_RESULT.BRANCHED}

    def branchexecps(self, allowaddcons):
        return {"result": SCIP_RESULT.DIDNOTRUN}


class MyBranchruleLocal(Branchrule):
    """
    A branching rule that tests addConsLocal by adding constraints to the current node.
    """

    def __init__(self, model):
        self.model = model
        self.addConsLocal_called = False
        self.call_count = 0

    def branchexeclp(self, allowaddcons):
        if not allowaddcons:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        self.call_count += 1

        # Only test on the first call
        if self.call_count > 1:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Get branching candidates
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.model.getLPBranchCands()

        if npriocands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        v = self.model.getVars()[0]
        cons = self.model.addConsLocal(v <= v.getLbGlobal() - 1)
        self.addConsLocal_called = True
        assert cons is not None, "addConsLocal should return a Constraint"

        return {"result": SCIP_RESULT.BRANCHED}
    
    def branchexecps(self, allowaddcons):
        return {"result": SCIP_RESULT.DIDNOTRUN}


def test_addConsNode():
    """Test that addConsNode works with ExprCons."""
    m = random_mip_1(node_lim=3, small=True)

    branchrule = MyBranchrule(m)
    m.includeBranchrule(
        branchrule,
        "test_addConsNode",
        "test addConsNode with ExprCons",
        priority=10000000,
        maxdepth=-1,
        maxbounddist=1
    )

    var_to_be_branched = m.getVars()[0]
    var_to_be_branched_lb = var_to_be_branched.getLbGlobal()

    m.optimize()

    assert branchrule.addConsNode_called, "addConsNode should have been called"

    var_to_be_branched_val = m.getSolVal(expr=var_to_be_branched, sol=None)
    assert var_to_be_branched_val == var_to_be_branched_lb, \
        f"Variable should be equal to its lower bound {var_to_be_branched_lb}, but got {var_to_be_branched_val}"
    


def test_addConsLocal():
    """Test that addConsLocal works with ExprCons."""
    m = random_mip_1(node_lim=500, small=True)

    branchrule = MyBranchruleLocal(m)
    m.includeBranchrule(
        branchrule,
        "test_addConsLocal",
        "test addConsLocal with ExprCons",
        priority=10000000,
        maxdepth=-1,
        maxbounddist=1
    )

    m.optimize()
    assert branchrule.addConsLocal_called, "addConsLocal should have been called"
    assert m.getStatus() == "infeasible", "The problem should be infeasible after adding the local constraint"
