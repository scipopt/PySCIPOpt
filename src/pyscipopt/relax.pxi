cdef class Relax:
    cdef public Model model
    cdef public str name

    def relaxfree(self):
        pass

    def relaxinit(self):
        pass

    def relaxexit(self):
        pass

    def relaxinitsol(self):
        pass

    def relaxexitsol(self):
        pass
        
    def relaxexec(self):
        print("python error in relaxexec: this method needs to be implemented")
        return{}


cdef SCIP_RETCODE PyRelaxCopy (SCIP* scip, SCIP_RELAX* relax):
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxFree (SCIP* scip, SCIP_RELAX* relax):
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxfree()
    Py_DECREF(PyRelax)
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxInit (SCIP* scip, SCIP_RELAX* relax):
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxExit (SCIP* scip, SCIP_RELAX* relax):
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxInitsol (SCIP* scip, SCIP_RELAX* relax):
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxExitsol (SCIP* scip, SCIP_RELAX* relax):
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxExec (SCIP* scip, SCIP_RELAX* relax, SCIP_Real* lowerbound, SCIP_RESULT* result):
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxexec()
    return SCIP_OKAY
    
