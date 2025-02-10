from pyscipopt import SCIP_RESULT, Eventhdlr, SCIP_EVENTTYPE
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