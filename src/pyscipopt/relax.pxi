##@file relax.pxi
#@brief Base class of the Relaxator Plugin
cdef class Relax:
    cdef public Model model
    cdef public str name

    def relaxfree(self):
        '''calls destructor and frees memory of relaxation handler'''
        pass

    def relaxinit(self):
        '''initializes relaxation handler'''
        pass

    def relaxexit(self):
        '''calls exit method of relaxation handler'''
        pass

    def relaxinitsol(self):
        '''informs relaxaton handler that the branch and bound process is being started'''
        pass

    def relaxexitsol(self):
        '''informs relaxation handler that the branch and bound process data is being freed'''
        pass
        
    def relaxexec(self):
        '''callls execution method of relaxation handler'''
        print("python error in relaxexec: this method needs to be implemented")
        return{}


cdef SCIP_RETCODE PyRelaxCopy (SCIP* scip, SCIP_RELAX* relax) with gil:
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxFree (SCIP* scip, SCIP_RELAX* relax) with gil:
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxfree()
    Py_DECREF(PyRelax)
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxInit (SCIP* scip, SCIP_RELAX* relax) with gil:
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxExit (SCIP* scip, SCIP_RELAX* relax) with gil:
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxInitsol (SCIP* scip, SCIP_RELAX* relax) with gil:
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxExitsol (SCIP* scip, SCIP_RELAX* relax) with gil:
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxexitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxExec (SCIP* scip, SCIP_RELAX* relax, SCIP_Real* lowerbound, SCIP_RESULT* result) with gil:
    cdef SCIP_RELAXDATA* relaxdata
    relaxdata = SCIPrelaxGetData(relax)
    PyRelax = <Relax>relaxdata
    PyRelax.relaxexec()
    return SCIP_OKAY
    
