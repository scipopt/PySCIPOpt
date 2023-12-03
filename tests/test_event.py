import pytest, random

from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING, quicksum

calls = []

class MyEvent(Eventhdlr):

    def eventinit(self):
        calls.append('eventinit')
        #self.model.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)
        self.model.catchEvent(self.event_type, self)        

    def eventexit(self):
        self.model.dropEvent(self.event_type, self)

    def eventexec(self, event):
        calls.append('eventexec')
        print(event.getType(),self.event_type)
        if self.event_type == SCIP_EVENTTYPE.LPEVENT:
            assert event.getType() in [SCIP_EVENTTYPE.FIRSTLPSOLVED, SCIP_EVENTTYPE.LPSOLVED]
        else:
            assert event.getType() == self.event_type
        #assert event.getNode().getNumber() == 1
        #assert event.getNode().getParent() is None


def test_event():

    all_events = [SCIP_EVENTTYPE.DISABLED,SCIP_EVENTTYPE.VARADDED,SCIP_EVENTTYPE.VARDELETED,SCIP_EVENTTYPE.VARFIXED,SCIP_EVENTTYPE.VARUNLOCKED,SCIP_EVENTTYPE.OBJCHANGED,SCIP_EVENTTYPE.GLBCHANGED,SCIP_EVENTTYPE.GUBCHANGED,SCIP_EVENTTYPE.LBTIGHTENED,SCIP_EVENTTYPE.LBRELAXED,SCIP_EVENTTYPE.UBTIGHTENED,SCIP_EVENTTYPE.UBRELAXED,SCIP_EVENTTYPE.GHOLEADDED,SCIP_EVENTTYPE.GHOLEREMOVED,SCIP_EVENTTYPE.LHOLEADDED,SCIP_EVENTTYPE.LHOLEREMOVED,SCIP_EVENTTYPE.IMPLADDED,SCIP_EVENTTYPE.PRESOLVEROUND,SCIP_EVENTTYPE.NODEFOCUSED,SCIP_EVENTTYPE.NODEFEASIBLE,SCIP_EVENTTYPE.NODEINFEASIBLE,SCIP_EVENTTYPE.NODEBRANCHED,SCIP_EVENTTYPE.FIRSTLPSOLVED,SCIP_EVENTTYPE.LPSOLVED,SCIP_EVENTTYPE.POORSOLFOUND,SCIP_EVENTTYPE.BESTSOLFOUND,SCIP_EVENTTYPE.ROWADDEDSEPA,SCIP_EVENTTYPE.ROWDELETEDSEPA,SCIP_EVENTTYPE.ROWADDEDLP,SCIP_EVENTTYPE.ROWDELETEDLP,SCIP_EVENTTYPE.ROWCOEFCHANGED,SCIP_EVENTTYPE.ROWCONSTCHANGED,SCIP_EVENTTYPE.ROWSIDECHANGED,SCIP_EVENTTYPE.SYNC,SCIP_EVENTTYPE.LPEVENT]
    
    all_event_hdlrs = []
    for event in all_events:
        # create solver instance
        s = Model()
        s.hideOutput()
        s.setPresolve(SCIP_PARAMSETTING.OFF)
        all_event_hdlrs.append(MyEvent())
        all_event_hdlrs[-1].event_type = event
        s.includeEventhdlr(all_event_hdlrs[-1], str(event), "python event handler to catch %s" % str(event))

        x = {}
        # add some variables
        for i in range(100):
            x[i] = s.addVar("x", obj=random.random(), vtype="I")

        # add some constraints
        for j in range(1,20):
            s.addCons(quicksum(x[i] for i in range(100) if i%j==0) >= random.randint(10,100))

        # solve problem
        s.optimize()

    del s

test_event()