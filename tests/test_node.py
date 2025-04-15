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

        return {'result': SCIP_RESULT.SUCCESS}
 
def test_getSiblings():
    m = random_mip_1(disable_heur=True, disable_presolve=True, disable_sepa=True)
    m.setParam("limits/nodes", 10)

    hdlr = focusEventHdlr()
    
    m.includeEventhdlr(hdlr, "test", "test")

    m.optimize()

    assert m.getNSols() == 0