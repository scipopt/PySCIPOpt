import pytest, random

from pyscipopt import Model, Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE, SCIP_PARAMSETTING, quicksum

calls = []
var_events = [SCIP_EVENTTYPE.VARCHANGED, 
              SCIP_EVENTTYPE.VAREVENT,
              SCIP_EVENTTYPE.BOUNDCHANGED,
              SCIP_EVENTTYPE.BOUNDRELAXED,
              SCIP_EVENTTYPE.BOUNDTIGHTENED,
              SCIP_EVENTTYPE.LBCHANGED,
              SCIP_EVENTTYPE.DOMCHANGED,
              SCIP_EVENTTYPE.GBDCHANGED,
              SCIP_EVENTTYPE.GHOLEADDED, 
              SCIP_EVENTTYPE.GHOLECHANGED, 
              SCIP_EVENTTYPE.GHOLEREMOVED, 
              SCIP_EVENTTYPE.GLBCHANGED, 
              SCIP_EVENTTYPE.GUBCHANGED, 
              SCIP_EVENTTYPE.HOLECHANGED, 
              SCIP_EVENTTYPE.IMPLADDED, 
              SCIP_EVENTTYPE.LBCHANGED, 
              SCIP_EVENTTYPE.LBRELAXED, 
              SCIP_EVENTTYPE.LBTIGHTENED, 
              SCIP_EVENTTYPE.LHOLEADDED, 
              SCIP_EVENTTYPE.LHOLECHANGED, 
              SCIP_EVENTTYPE.LHOLEREMOVED, 
              SCIP_EVENTTYPE.OBJCHANGED, 
              SCIP_EVENTTYPE.UBCHANGED, 
              SCIP_EVENTTYPE.UBRELAXED, 
              SCIP_EVENTTYPE.UBTIGHTENED, 
              SCIP_EVENTTYPE.VARDELETED, 
              SCIP_EVENTTYPE.VARFIXED, 
              SCIP_EVENTTYPE.VARUNLOCKED]

row_events = [SCIP_EVENTTYPE.ROWCHANGED,
              SCIP_EVENTTYPE.ROWADDEDLP,
              SCIP_EVENTTYPE.ROWADDEDSEPA, 
              SCIP_EVENTTYPE.ROWCHANGED, 
              SCIP_EVENTTYPE.ROWCOEFCHANGED,
              SCIP_EVENTTYPE.ROWCONSTCHANGED,
              SCIP_EVENTTYPE.ROWDELETEDLP, 
              SCIP_EVENTTYPE.ROWDELETEDSEPA, 
              SCIP_EVENTTYPE.ROWEVENT, 
              SCIP_EVENTTYPE.ROWSIDECHANGED]

class MyEvent(Eventhdlr):
    def __init__(self):
        super().__init__()
        self.event_type = None

    def eventinit(self):
        calls.append('eventinit')
        print("init ", self.event_type)

        if self.event_type in [SCIP_EVENTTYPE.VAREVENT, SCIP_EVENTTYPE.VARCHANGED]:
            print(f"Skipping composite event type: {self.event_type}")
            return
        
        if self.event_type in var_events:
            var = self.model.getTransformedVar(self.model.getVars()[0])
            self.model.catchVarEvent(var, self.event_type, self)
        elif self.event_type in row_events:
            print(str(self.event_type), " requires row")
            pass
            # self.model.catchRowEvent(row, self.event_type, self)
        else:
            self.model.catchEvent(self.event_type, self)        

    def eventexit(self):
        # PR #828 fixes an error here, but the underlying cause might not be solved (self.model being deleted before dropEvent is called)
        # self.model.dropEvent(self.event_type, self) # <- gives an UnraisableExceptionWarning: weakly-referenced object no longer exists
        pass

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
            assert event.getType() in [SCIP_EVENTTYPE.LBCHANGED, SCIP_EVENTTYPE.UBCHANGED, SCIP_EVENTTYPE.UBTIGHTENED, SCIP_EVENTTYPE.UBRELAXED, SCIP_EVENTTYPE.LBTIGHTENED, SCIP_EVENTTYPE.LBRELAXED]
        elif self.event_type == SCIP_EVENTTYPE.GHOLECHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.GHOLEADDED, SCIP_EVENTTYPE.GHOLEREMOVED]
        elif self.event_type == SCIP_EVENTTYPE.LHOLECHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.LHOLEADDED, SCIP_EVENTTYPE.LHOLEREMOVED]
        elif self.event_type == SCIP_EVENTTYPE.HOLECHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.GHOLECHANGED, SCIP_EVENTTYPE.LHOLECHANGED]
        elif self.event_type == SCIP_EVENTTYPE.DOMCHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.BOUNDCHANGED, SCIP_EVENTTYPE.HOLECHANGED, SCIP_EVENTTYPE.UBTIGHTENED, SCIP_EVENTTYPE.UBRELAXED, SCIP_EVENTTYPE.LBTIGHTENED, SCIP_EVENTTYPE.LBRELAXED]    
        elif self.event_type == SCIP_EVENTTYPE.NODESOLVED:
            assert event.getType() in [SCIP_EVENTTYPE.NODEFEASIBLE, SCIP_EVENTTYPE.NODEINFEASIBLE, SCIP_EVENTTYPE.NODEBRANCHED]
        elif self.event_type == SCIP_EVENTTYPE.NODEEVENT:
            assert event.getType() in [SCIP_EVENTTYPE.NODEFOCUSED, SCIP_EVENTTYPE.NODEFEASIBLE, SCIP_EVENTTYPE.NODEINFEASIBLE, SCIP_EVENTTYPE.NODEBRANCHED]
        elif self.event_type == SCIP_EVENTTYPE.LPEVENT:
            assert event.getType() in [SCIP_EVENTTYPE.FIRSTLPSOLVED, SCIP_EVENTTYPE.LPSOLVED]
        elif self.event_type == SCIP_EVENTTYPE.SOLFOUND:
            assert event.getType() in [SCIP_EVENTTYPE.POORSOLFOUND, SCIP_EVENTTYPE.BESTSOLFOUND]        
        elif self.event_type == SCIP_EVENTTYPE.ROWCHANGED:
            assert event.getType() in [SCIP_EVENTTYPE.ROWCOEFCHANGED, SCIP_EVENTTYPE.ROWCONSTCHANGED, SCIP_EVENTTYPE.ROWSIDECHANGED]
        elif self.event_type == SCIP_EVENTTYPE.ROWEVENT:
            assert event.getType() in [SCIP_EVENTTYPE.ROWADDEDSEPA, SCIP_EVENTTYPE.ROWDELETEDSEPA, SCIP_EVENTTYPE.ROWADDEDLP, SCIP_EVENTTYPE.ROWDELETEDLP, SCIP_EVENTTYPE.ROWCHANGED]
        elif self.event_type == SCIP_EVENTTYPE.GAPUPDATED:
            assert event.getType() in [SCIP_EVENTTYPE.DUALBOUNDIMPROVED, SCIP_EVENTTYPE.BESTSOLFOUND]
        else:
            pass

def test_event():

    all_events = {}
    for attr_name in dir(SCIP_EVENTTYPE):
        if not attr_name.startswith('_'):
            attr = getattr(SCIP_EVENTTYPE, attr_name)
            if isinstance(attr, int):
                all_events[attr_name] = attr
    
    all_event_hdlrs = []
    for event_name, event in all_events.items():
        s = Model()
        s.hideOutput()
        all_event_hdlrs.append(MyEvent())
        all_event_hdlrs[-1].event_type = all_events[event_name]
        s.includeEventhdlr(all_event_hdlrs[-1], str(event), "python event handler to catch %s" % str(event_name))

        x = {}
        for i in range(100):
            x[i] = s.addVar("x", obj=random.random(), vtype="I")

        for j in range(1,20):
            s.addCons(quicksum(x[i] for i in range(100) if i%j==0) >= random.randint(10,100))

        if event in var_events or event in row_events:
            s.presolve()
        else:
            s.setPresolve(SCIP_PARAMSETTING.OFF)
            all_event_hdlrs[-1].var = None

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

def test_raise_error_catch_var_event():
    m = Model()
    m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    
    class MyEventVar(Eventhdlr):
        def __init__(self, var):
            super().__init__()
            self.var = var

        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.VAREVENT, self)        

        def eventexit(self):
            pass
            # self.model..dropEvent(SCIP_EVENTTYPE.VAREVENT, self)

        def eventexec(self, event):
            pass

    v = m.addVar("x", vtype="I")
    ev = MyEventVar(v)
    m.includeEventhdlr(ev, "var_event", "event handler for var events")

    with pytest.raises(Exception):
        m.optimize()