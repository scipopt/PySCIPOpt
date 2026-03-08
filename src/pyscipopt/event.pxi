##@file event.pxi
#@brief Base class of the Event Handler Plugin
cdef class Eventhdlr:
    cdef public Model model
    cdef public str name
    cdef public list _caught_events

    def __cinit__(self):
        self._caught_events = []

    def eventcopy(self):
        '''sets copy callback for all events of this event handler '''
        pass

    def eventfree(self):
        '''calls destructor and frees memory of event handler '''
        pass

    def eventinit(self):
        '''initializes event handler'''
        pass

    def eventexit(self):
        '''calls exit method of event handler'''
        pass

    def eventinitsol(self):
        '''informs event handler that the branch and bound process is being started '''
        pass

    def eventexitsol(self):
        '''informs event handler that the branch and bound process data is being freed '''
        pass

    def eventdelete(self):
        '''sets callback to free specific event data'''
        pass

    def eventexec(self, event):
        '''calls execution method of event handler '''
        raise NotImplementedError("eventexec() is a fundamental callback and should be implemented in the derived class")


# local helper functions for the interface
cdef Eventhdlr getPyEventhdlr(SCIP_EVENTHDLR* eventhdlr):
    cdef SCIP_EVENTHDLRDATA* eventhdlrdata
    eventhdlrdata = SCIPeventhdlrGetData(eventhdlr)
    return <Eventhdlr>eventhdlrdata

cdef SCIP_RETCODE PyEventCopy (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) noexcept with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventcopy()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventFree (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) noexcept with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventfree()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventInit (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) noexcept with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExit (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) noexcept with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventexit()
    # Auto-drop any events not explicitly dropped by the user
    for eventtype in PyEventhdlr._caught_events:
        SCIPdropEvent(scip, eventtype, eventhdlr, NULL, -1)
    PyEventhdlr._caught_events = []
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventInitsol (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) noexcept with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExitsol (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) noexcept with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventDelete (SCIP* scip, SCIP_EVENTHDLR* eventhdlr, SCIP_EVENTDATA** eventdata) noexcept with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventdelete()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExec (SCIP* scip, SCIP_EVENTHDLR* eventhdlr, SCIP_EVENT* event, SCIP_EVENTDATA* eventdata) noexcept with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEvent = Event()
    PyEvent.event = event
    PyEventhdlr.eventexec(PyEvent)
    return SCIP_OKAY
