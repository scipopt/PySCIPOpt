import pytest, random

from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING, quicksum

calls = []


class MyEvent(Eventhdlr):

    def eventinit(self):
        calls.append('eventinit')

        self._tracked_vars = []
        self._tracked_rows = []
        if self.event_type & SCIP_EVENTTYPE.VARCHANGED:
            vars = self.model.getVars(transformed=True)
            if not vars:
                vars = [self.model.getTransformedVar(var) for var in self.model.getVars()]
            for var in vars:
                self.model.catchVarEvent(var, self.event_type, self)
                self._tracked_vars.append(var)
        elif self.event_type & SCIP_EVENTTYPE.ROWCHANGED:
            try:
                if self.model.getNLPRows() == 0:
                    return
                rows = self.model.getLPRowsData()
            except Exception:
                return
            for row in rows:
                self.model.catchRowEvent(row, self.event_type, self)
                self._tracked_rows.append(row)
        else:
            self.model.catchEvent(self.event_type, self)

    def eventexit(self):
        # PR #828 fixes an error here, but the underlying cause might not be solved (self.model being deleted before dropEvent is called)
        if self.event_type & SCIP_EVENTTYPE.VARCHANGED:
            for var in self._tracked_vars:
                try:
                    self.model.dropVarEvent(var, self.event_type, self)
                except ReferenceError:
                    pass
        elif self.event_type & SCIP_EVENTTYPE.ROWCHANGED:
            for row in self._tracked_rows:
                try:
                    self.model.dropRowEvent(row, self.event_type, self)
                except ReferenceError:
                    pass
        else:
            self.model.dropEvent(self.event_type, self)

    def eventexec(self, event):
        assert str(event) == event.getName()
        assert type(event.getName()) == str

        calls.append('eventexec')
        if self.event_type == SCIP_EVENTTYPE.LPEVENT:
            assert event.getType() in [SCIP_EVENTTYPE.FIRSTLPSOLVED, SCIP_EVENTTYPE.LPSOLVED]
        elif self.event_type == SCIP_EVENTTYPE.GBDCHANGED:    
            assert event.getType() in [SCIP_EVENTTYPE.GLBCHANGED, SCIP_EVENTTYPE.GUBCHANGED]
        elif self.event_type == SCIP_EVENTTYPE.LBCHANGED:     
            assert event.getType() in [SCIP_EVENTTYPE.LBTIGHTENED, SCIP_EVENTTYPE.LBRELAXED]
        elif self.event_type == SCIP_EVENTTYPE.UBCHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.UBTIGHTENED, SCIP_EVENTTYPE.UBRELAXED]
        elif self.event_type == SCIP_EVENTTYPE.BOUNDTIGHTENED:
            assert event.getType() in [SCIP_EVENTTYPE.LBTIGHTENED, SCIP_EVENTTYPE.UBTIGHTENED]
        elif self.event_type == SCIP_EVENTTYPE.BOUNDRELAXED:
            assert event.getType() in [SCIP_EVENTTYPE.LBRELAXED, SCIP_EVENTTYPE.UBRELAXED]
        elif self.event_type == SCIP_EVENTTYPE.BOUNDCHANGED:
            assert event.getType() & SCIP_EVENTTYPE.BOUNDCHANGED
        elif self.event_type == SCIP_EVENTTYPE.GHOLECHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.GHOLEADDED, SCIP_EVENTTYPE.GHOLEREMOVED]
        elif self.event_type == SCIP_EVENTTYPE.LHOLECHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.LHOLEADDED, SCIP_EVENTTYPE.LHOLEREMOVED]
        elif self.event_type == SCIP_EVENTTYPE.HOLECHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.GHOLECHANGED, SCIP_EVENTTYPE.LHOLECHANGED]
        elif self.event_type == SCIP_EVENTTYPE.DOMCHANGED:
            assert event.getType() & SCIP_EVENTTYPE.DOMCHANGED
        elif self.event_type == SCIP_EVENTTYPE.VARCHANGED:
            assert event.getType() & SCIP_EVENTTYPE.VARCHANGED
        elif self.event_type == SCIP_EVENTTYPE.VAREVENT:
            assert event.getType() & SCIP_EVENTTYPE.VAREVENT
        elif self.event_type == SCIP_EVENTTYPE.NODESOLVED:
            assert event.getType() in [SCIP_EVENTTYPE.NODEFEASIBLE, SCIP_EVENTTYPE.NODEINFEASIBLE, SCIP_EVENTTYPE.NODEBRANCHED]
        elif self.event_type == SCIP_EVENTTYPE.NODEEVENT:
            assert event.getType() in [SCIP_EVENTTYPE.NODEFOCUSED, SCIP_EVENTTYPE.NODEFEASIBLE, SCIP_EVENTTYPE.NODEINFEASIBLE, SCIP_EVENTTYPE.NODEBRANCHED]
        elif self.event_type == SCIP_EVENTTYPE.LPEVENT:
            assert event.getType() in [SCIP_EVENTTYPE.FIRSTLPSOLVED, SCIP_EVENTTYPE.LPSOLVED]
        elif self.event_type == SCIP_EVENTTYPE.SOLFOUND:
            assert event.getType() in [SCIP_EVENTTYPE.POORSOLFOUND, SCIP_EVENTTYPE.BESTSOLFOUND]        
        elif self.event_type == SCIP_EVENTTYPE.ROWCHANGED:
            assert event.getType() & SCIP_EVENTTYPE.ROWCHANGED
        elif self.event_type == SCIP_EVENTTYPE.ROWEVENT:
            assert event.getType() & SCIP_EVENTTYPE.ROWEVENT
        else:
            assert event.getType() == self.event_type

def test_event():

    all_events = [SCIP_EVENTTYPE.DISABLED,SCIP_EVENTTYPE.VARADDED,SCIP_EVENTTYPE.VARDELETED,SCIP_EVENTTYPE.VARFIXED,SCIP_EVENTTYPE.VARUNLOCKED,SCIP_EVENTTYPE.OBJCHANGED,SCIP_EVENTTYPE.GLBCHANGED,SCIP_EVENTTYPE.GUBCHANGED,SCIP_EVENTTYPE.LBTIGHTENED,SCIP_EVENTTYPE.LBRELAXED,SCIP_EVENTTYPE.UBTIGHTENED,SCIP_EVENTTYPE.UBRELAXED,SCIP_EVENTTYPE.GHOLEADDED,SCIP_EVENTTYPE.GHOLEREMOVED,SCIP_EVENTTYPE.LHOLEADDED,SCIP_EVENTTYPE.LHOLEREMOVED,SCIP_EVENTTYPE.IMPLADDED,SCIP_EVENTTYPE.PRESOLVEROUND,SCIP_EVENTTYPE.NODEFOCUSED,SCIP_EVENTTYPE.NODEFEASIBLE,SCIP_EVENTTYPE.NODEINFEASIBLE,SCIP_EVENTTYPE.NODEBRANCHED,SCIP_EVENTTYPE.NODEDELETE,SCIP_EVENTTYPE.FIRSTLPSOLVED,SCIP_EVENTTYPE.LPSOLVED,SCIP_EVENTTYPE.POORSOLFOUND,SCIP_EVENTTYPE.BESTSOLFOUND,SCIP_EVENTTYPE.ROWADDEDSEPA,SCIP_EVENTTYPE.ROWDELETEDSEPA,SCIP_EVENTTYPE.ROWADDEDLP,SCIP_EVENTTYPE.ROWDELETEDLP,SCIP_EVENTTYPE.ROWCOEFCHANGED,SCIP_EVENTTYPE.ROWCONSTCHANGED,SCIP_EVENTTYPE.ROWSIDECHANGED,SCIP_EVENTTYPE.SYNC,SCIP_EVENTTYPE.GBDCHANGED,SCIP_EVENTTYPE.LBCHANGED,SCIP_EVENTTYPE.UBCHANGED,SCIP_EVENTTYPE.BOUNDTIGHTENED,SCIP_EVENTTYPE.BOUNDRELAXED,SCIP_EVENTTYPE.BOUNDCHANGED,SCIP_EVENTTYPE.LHOLECHANGED,SCIP_EVENTTYPE.HOLECHANGED,SCIP_EVENTTYPE.DOMCHANGED,SCIP_EVENTTYPE.VARCHANGED,SCIP_EVENTTYPE.VAREVENT,SCIP_EVENTTYPE.NODESOLVED,SCIP_EVENTTYPE.NODEEVENT,SCIP_EVENTTYPE.LPEVENT,SCIP_EVENTTYPE.SOLFOUND,SCIP_EVENTTYPE.SOLEVENT,SCIP_EVENTTYPE.ROWCHANGED,SCIP_EVENTTYPE.ROWEVENT]

    all_event_hdlrs = []
    for event in all_events:
        s = Model()
        s.hideOutput()
        s.setPresolve(SCIP_PARAMSETTING.OFF)
        all_event_hdlrs.append(MyEvent())
        all_event_hdlrs[-1].event_type = event
        s.includeEventhdlr(all_event_hdlrs[-1], str(event), "python event handler to catch %s" % str(event))

        x = {}
        for i in range(100):
            x[i] = s.addVar("x", obj=random.random(), vtype="I")

        for j in range(1,20):
            s.addCons(quicksum(x[i] for i in range(100) if i%j==0) >= random.randint(10,100))

        s.optimize()

def test_event_handler_callback(): 
    m = Model()
    m.hideOutput()
    
    number_of_calls = 0
    
    def callback(model, event):
        nonlocal number_of_calls
        number_of_calls += 1
        
    m.attachEventHandlerCallback(callback, [SCIP_EVENTTYPE.BESTSOLFOUND])
    m.attachEventHandlerCallback(callback, [SCIP_EVENTTYPE.BESTSOLFOUND])

    m.optimize()
    
    assert number_of_calls == 2
