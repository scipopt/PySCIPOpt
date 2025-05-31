##@file iisfinder.pxi
#@brief Base class of the Relaxator Plugin
cdef class IISFinder:
    cdef public Model model
    cdef public str name

    def iisfinderfree(self):
        '''calls destructor and frees memory of iis finder'''
        pass
        
    def iisfinderexec(self):
        '''calls execution method of iis finder'''
        raise NotImplementedError("iisfinderexec() is a fundamental callback and should be implemented in the derived class")
        

cdef SCIP_RETCODE PyIISFinderCopy (SCIP* scip, SCIP_IISFINDER* iisfinder) noexcept with gil:
    return SCIP_OKAY

cdef SCIP_RETCODE PyIISFinderFree (SCIP* scip, SCIP_IISFINDER* iisfinder) noexcept with gil:
    cdef SCIP_IISFINDERDATA* iisfinderdata
    iisfinderdata = SCIPIISfinderGetData(iisfinder)
    PyRelax = <Relax>iisfinderdata
    PyRelax.iisfinderfree()
    Py_DECREF(PyRelax)
    return SCIP_OKAY

cdef SCIP_RETCODE PyRelaxExec (SCIP* scip, SCIP_IISFINDER* iisfinder, SCIP_Real* lowerbound, SCIP_RESULT* result) noexcept with gil:
    cdef SCIP_IISFINDERDATA* iisfinderdata
    iisfinderdata = SCIPiisfinderGetData(iisfinder)
    PyRelax = <Relax>iisfinderdata
    result_dict = PyRelax.iisfinderexec()
    assert isinstance(result_dict, dict), "iisfinderexec() must return a dictionary."
    #TODO
    assert False
    lowerbound[0] = result_dict.get("lowerbound", <SCIP_Real>lowerbound[0])
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY