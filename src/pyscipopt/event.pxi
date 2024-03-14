##@file event.pxi
#@brief Base class of the Event Handler Plugin
cdef class Eventhdlr:
    cdef public Model model
    cdef public str name

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
        print("python error in eventexec: this method needs to be implemented")
        return {}


# local helper functions for the interface
cdef Eventhdlr getPyEventhdlr(SCIP_EVENTHDLR* eventhdlr) with gil:
    cdef SCIP_EVENTHDLRDATA* eventhdlrdata
    eventhdlrdata = SCIPeventhdlrGetData(eventhdlr)
    return <Eventhdlr>eventhdlrdata

cdef SCIP_RETCODE PyEventCopy (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventcopy()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventFree (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventfree()
    Py_DECREF(PyEventhdlr)
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventInit (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExit (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventInitsol (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExitsol (SCIP* scip, SCIP_EVENTHDLR* eventhdlr) with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventDelete (SCIP* scip, SCIP_EVENTHDLR* eventhdlr, SCIP_EVENTDATA** eventdata) with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEventhdlr.eventdelete()
    return SCIP_OKAY

cdef SCIP_RETCODE PyEventExec (SCIP* scip, SCIP_EVENTHDLR* eventhdlr, SCIP_EVENT* event, SCIP_EVENTDATA* eventdata) with gil:
    PyEventhdlr = getPyEventhdlr(eventhdlr)
    PyEvent = Event()
    PyEvent.event = event
    PyEventhdlr.eventexec(PyEvent)
    return SCIP_OKAY
