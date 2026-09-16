from pyscipopt import SCIP_RESULT, Eventhdlr, SCIP_EVENTTYPE, scip
from helpers.utils import random_mip_1

class cutoffEventHdlr(Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        self.model.cutoffNode(self.model.getCurrentNode())
        return {'result': SCIP_RESULT.SUCCESS}

def test_cutoffNode():
    m = random_mip_1(disable_heur=True, disable_presolve=True, disable_sepa=True)
    
    hdlr = cutoffEventHdlr()
    
    m.includeEventhdlr(hdlr, "test", "test")

    m.optimize()

    assert m.getNSols() == 0

class focusEventHdlr(Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        assert self.model.getNSiblings() in [0,1]
        assert len(self.model.getSiblings()) == self.model.getNSiblings()
        for node in self.model.getSiblings():
            assert isinstance(node, scip.Node)

        assert self.model.getNLeaves() >= 0
        assert len(self.model.getLeaves()) == self.model.getNLeaves()
        for node in self.model.getLeaves():
            assert isinstance(node, scip.Node)
        
        assert self.model.getNChildren() >= 0
        assert len(self.model.getChildren()) == self.model.getNChildren()
        for node in self.model.getChildren():
            assert isinstance(node, scip.Node)

        leaves, children, siblings = self.model.getOpenNodes()
        assert leaves == self.model.getLeaves()
        assert children == self.model.getChildren()
        assert siblings == self.model.getSiblings()

        nodes_left = self.model.getNNodesLeft()
        assert (
            nodes_left
            == self.model.getNLeaves()
            + self.model.getNChildren()
            + self.model.getNSiblings()
        )   

        return {'result': SCIP_RESULT.SUCCESS}
 
def test_tree_methods():
    m = random_mip_1(disable_heur=True, disable_presolve=True, disable_sepa=True)
    m.setParam("limits/nodes", 10)

    hdlr = focusEventHdlr()
    
    m.includeEventhdlr(hdlr, "test", "test")

    m.optimize()

    assert m.getNSols() == 0

class ProbingNodeChecker(Eventhdlr):
    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        m = self.model
        focus = m.getFocusNode()
        current = m.getCurrentNode()

        assert isinstance(focus, scip.Node)
        assert isinstance(current, scip.Node)

        # focus and current node should be the same before probing
        assert focus.getNumber() == current.getNumber()

        m.startProbing()
        m.newProbingNode()
        m.newProbingNode()

        # after starting probing, the focus node should remain the same, but the current node should change
        assert m.getProbingDepth() == 2
        assert m.getFocusNode().getNumber() == focus.getNumber()
        assert m.getCurrentNode().getNumber() != current.getNumber()

        m.endProbing()

        return {'result': SCIP_RESULT.SUCCESS}

def test_getFocusNode_and_getCurrentNode():

    m = random_mip_1(small=True)

    m.includeEventhdlr(ProbingNodeChecker(), "Probing Node Checker", "test if getFocusNode and getCurrentNode work correctly")
    m.optimize()
