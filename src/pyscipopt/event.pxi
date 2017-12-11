cdef class Eventhdlr:
    cdef public Model model
    cdef public str name

    def eventfree(self):
        pass

    def eventinit(self):
        pass

    def eventexit(self):
        pass

    def eventinitsol(self):
        pass

    def eventexitsol(self):
        pass

    def eventdelete(self):
        pass

    def eventexec(self, event):
        pass


# local helper functions for the interface
cdef Eventhdlr getPyEventhdlr(SCIP_EVENTHDLR* eventhdlr):
    cdef SCIP_EVENTHDLRDATA* eventhdlrdata
    eventhdlrdata = SCIPeventhdlrGetData(eventhdlr)
    return <Eventhdlr>eventhdlrdata

cdef SCIP_RETCODE PyEventCopy (SCIP* scip, SCIP_EVENTHDLR* eventhdlr):
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventFree (SCIP* scip, SCIP_EVENTHDLR* eventhdlr):
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventfree()
    Py_DECREF(PyEventhdlr)
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventInit (SCIP* scip, SCIP_EVENTHDLR* eventhdlr):
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExit (SCIP* scip, SCIP_EVENTHDLR* eventhdlr):
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventInitsol (SCIP* scip, SCIP_EVENTHDLR* eventhdlr):
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExitsol (SCIP* scip, SCIP_EVENTHDLR* eventhdlr):
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventDelete (SCIP* scip, SCIP_EVENTHDLR* eventhdlr, SCIP_EVENTDATA** eventdata):
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventdelete()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExec (SCIP* scip, SCIP_EVENTHDLR* eventhdlr, SCIP_EVENT* event, SCIP_EVENTDATA* eventdata):
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEvent = Event()
    PyEvent.event = event
    PyEventhdlr.eventexeclp(PyEvent)
    return SCIP_OKAY
